//! Publishes where the panels are in the room, for programs that place their own content relative
//! to a screen (a browser extension that puts a 3D model where a page shows it flat). Poses are in
//! the stage space, as 4x4 matrices in column order; mirrored screens carry their place on the
//! desktop. Updated about ten times a second, on request:
//!
//! - `WAYVR_LAYOUT_PORT=<port>`: served as JSON over HTTP on 127.0.0.1:<port> (GET, any path).
//! - `WAYVR_LAYOUT_FILE=1`: written to `$XDG_RUNTIME_DIR/wayvr-layout.json`.
//!
//! The same port takes the other direction: `POST /regions` with `{"regions": [{id, rect, screen?,
//! pose?}]}` says which rectangles of a mirrored screen are to be panels of their own (see
//! `overlays::region`). The list is complete each time: regions missing from it are removed.

use std::{
    collections::HashMap,
    io::{Read, Write},
    net::TcpListener,
    path::PathBuf,
    sync::{Arc, Mutex, OnceLock},
    time::Duration,
};

use glam::{Affine3A, Mat4};
use serde_json::json;

use super::overlay::OpenXrOverlayData;
use crate::{
    backend::task::{OverlayTask, SpawnPos, TaskType},
    overlays::region::{self, RegionSpec, SharedSpec},
    state::AppState,
    windowing::{OverlaySelector, manager::OverlayWindowManager},
};

struct Outputs {
    file: Option<PathBuf>,
    http: Option<Arc<Mutex<String>>>,
}

fn serve(port: u16, doc: Arc<Mutex<String>>) {
    let listener = match TcpListener::bind(("127.0.0.1", port)) {
        Ok(l) => l,
        Err(e) => {
            log::warn!("layout: cannot listen on 127.0.0.1:{port}: {e}");
            return;
        }
    };
    log::info!("layout: serving the panel layout on http://127.0.0.1:{port}/");
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let _ = stream.set_read_timeout(Some(Duration::from_millis(200)));
            let _ = stream.set_write_timeout(Some(Duration::from_millis(200)));
            let mut req = Vec::new();
            let mut buf = [0u8; 4096];
            // headers, then as much body as Content-Length says (regions are small)
            let mut want: Option<usize> = None;
            while req.len() < 256 * 1024 {
                let Ok(n) = stream.read(&mut buf) else { break };
                if n == 0 {
                    break;
                }
                req.extend_from_slice(&buf[..n]);
                if want.is_none()
                    && let Some(end) = req.windows(4).position(|w| w == b"\r\n\r\n")
                {
                    let head = String::from_utf8_lossy(&req[..end]).to_ascii_lowercase();
                    let len = head
                        .lines()
                        .find_map(|l| l.strip_prefix("content-length:"))
                        .and_then(|v| v.trim().parse::<usize>().ok())
                        .unwrap_or(0);
                    want = Some(end + 4 + len);
                }
                if want.is_some_and(|w| req.len() >= w) {
                    break;
                }
            }
            if req.starts_with(b"POST /regions")
                && let Some(end) = req.windows(4).position(|w| w == b"\r\n\r\n")
                && let Ok(mut inbox) = REGION_INBOX.lock()
            {
                *inbox = Some(String::from_utf8_lossy(&req[end + 4..]).into_owned());
            }
            let body = doc.lock().map(|d| d.clone()).unwrap_or_default();
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(head.as_bytes());
            let _ = stream.write_all(body.as_bytes());
        }
    });
}

/// The newest region list received, until the frame loop takes it.
static REGION_INBOX: Mutex<Option<String>> = Mutex::new(None);

#[derive(serde::Deserialize)]
struct RegionList {
    regions: Vec<RegionSpec>,
}

thread_local! {
    /// Regions that have a panel: id -> shared rectangle, and the pose applied last.
    static REGIONS: std::cell::RefCell<HashMap<String, (SharedSpec, Option<Vec<f32>>)>> =
        std::cell::RefCell::new(HashMap::new());
}

/// Makes the panels follow the list: new regions get a panel, changed ones a new rectangle or
/// place, missing ones are removed.
fn apply_regions(
    app: &mut AppState,
    overlays: &OverlayWindowManager<OpenXrOverlayData>,
    text: &str,
) {
    let list: RegionList = match serde_json::from_str(text) {
        Ok(l) => l,
        Err(e) => {
            log::warn!("regions: {e}");
            return;
        }
    };
    // metres per logical desktop pixel, from the screen panels as they are now
    let density = |screen: Option<&str>| -> f32 {
        overlays
            .values()
            .filter(|o| screen.is_none_or(|n| &*o.config.name == n))
            .find_map(|o| {
                let rect = o.config.backend.desktop_rect()?;
                let state = o.config.active_state.as_ref()?;
                (rect[2] > 0.0).then(|| state.transform.matrix3.col(0).length() / rect[2])
            })
            .unwrap_or(0.0008)
    };
    REGIONS.with_borrow_mut(|known| {
        known.retain(|id, _| {
            let keep = list.regions.iter().any(|r| &r.id == id);
            if !keep {
                app.tasks.enqueue(TaskType::Overlay(OverlayTask::Drop(
                    OverlaySelector::Name(region::overlay_name(id)),
                )));
            }
            keep
        });
        for r in list.regions {
            let width_m = r.rect[2] * density(r.screen.as_deref());
            let pose = r.pose.clone().filter(|p| p.len() == 16);
            if let Some((spec, applied)) = known.get_mut(&r.id) {
                if let Ok(mut s) = spec.lock() {
                    *s = r.clone();
                }
                if pose.is_some() && *applied != pose {
                    *applied = pose.clone();
                    place(app, &r, width_m);
                }
            } else {
                let spec: SharedSpec = Arc::new(Mutex::new(r.clone()));
                known.insert(r.id.clone(), (spec.clone(), pose.clone()));
                app.tasks.enqueue(TaskType::Overlay(OverlayTask::Spawn(
                    OverlaySelector::Nothing,
                    SpawnPos::Spread,
                    Box::new(move |app: &mut AppState| {
                        region::create_region_overlay(app, spec, width_m)
                            .inspect_err(|e| log::warn!("region panel: {e:?}"))
                            .ok()
                    }),
                )));
                if pose.is_some() {
                    place(app, &r, width_m);
                }
            }
        }
    });
}

/// Puts a region's panel at the pose it asked for.
fn place(app: &mut AppState, r: &RegionSpec, width_m: f32) {
    let Some(p) = r.pose.as_ref().filter(|p| p.len() == 16) else {
        return;
    };
    let m = Mat4::from_cols_slice(p);
    let (_, rotation, translation) = m.to_scale_rotation_translation();
    let aspect = (r.rect[2] / r.rect[3].max(1.0)).max(0.01);
    let scale = if aspect < 1.0 { width_m / aspect } else { width_m };
    let transform =
        Affine3A::from_scale_rotation_translation(glam::Vec3::splat(scale), rotation, translation);
    app.tasks.enqueue(TaskType::Overlay(OverlayTask::Modify(
        OverlaySelector::Name(region::overlay_name(&r.id)),
        Box::new(move |_app, owc| {
            if let Some(state) = owc.active_state.as_mut() {
                state.transform = transform;
            } else {
                owc.default_state.transform = transform;
            }
        }),
    )));
}

fn outputs() -> &'static Outputs {
    static OUT: OnceLock<Outputs> = OnceLock::new();
    OUT.get_or_init(|| {
        let file = if std::env::var("WAYVR_LAYOUT_FILE").is_ok_and(|v| v != "0") {
            std::env::var("XDG_RUNTIME_DIR")
                .ok()
                .map(|d| PathBuf::from(d).join("wayvr-layout.json"))
        } else {
            None
        };
        let http = std::env::var("WAYVR_LAYOUT_PORT")
            .ok()
            .and_then(|p| p.parse::<u16>().ok())
            .filter(|p| *p != 0)
            .map(|port| {
                let doc = Arc::new(Mutex::new(String::from("{}")));
                serve(port, doc.clone());
                doc
            });
        Outputs { file, http }
    })
}

fn mat(a: &Affine3A) -> Vec<f32> {
    Mat4::from(*a).to_cols_array().to_vec()
}

pub(super) fn tick(
    app: &mut AppState,
    overlays: &OverlayWindowManager<OpenXrOverlayData>,
    frame: usize,
) {
    let out = outputs();
    if out.http.is_some()
        && let Some(text) = REGION_INBOX.lock().ok().and_then(|mut i| i.take())
    {
        apply_regions(app, overlays, &text);
    }
    if (out.file.is_none() && out.http.is_none()) || frame % 6 != 0 {
        return;
    }
    let list: Vec<_> = overlays
        .values()
        .filter_map(|o| {
            let state = o.config.active_state.as_ref()?;
            let extent = o.data.swapchain.as_ref().map(|s| s.extent)?;
            Some(json!({
                "name": &*o.config.name,
                "category": format!("{:?}", o.config.category),
                "transform": mat(&state.transform),
                "extent": extent,
                "curvature": state.curvature,
                "alpha": state.alpha,
                "desktop": o.config.backend.desktop_rect(),
            }))
        })
        .collect();
    let doc = json!({
        "frame": frame,
        "hmd": mat(&app.input_state.hmd),
        "overlays": list,
    })
    .to_string();

    if let Some(path) = out.file.as_ref() {
        let tmp = path.with_extension("json.tmp");
        let ok = std::fs::File::create(&tmp)
            .and_then(|mut f| f.write_all(doc.as_bytes()))
            .and_then(|()| std::fs::rename(&tmp, path));
        if let Err(e) = ok {
            log::debug!("layout file: {e}");
        }
    }
    if let Some(http) = out.http.as_ref()
        && let Ok(mut d) = http.lock()
    {
        *d = doc;
    }
}
