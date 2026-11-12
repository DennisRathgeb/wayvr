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
    /// The screens the regions are cut from are not shown themselves while this is set.
    #[serde(default)]
    hide_screen: bool,
}

/// When the last region list arrived. A sender that has gone quiet (its page closed) must not
/// leave the screen hidden: after `REGION_TIMEOUT` the regions go and the screen comes back.
static REGION_SEEN: Mutex<Option<std::time::Instant>> = Mutex::new(None);
const REGION_TIMEOUT: Duration = Duration::from_secs(10);

/// Hides or shows the mirrored screens themselves (they keep updating either way).
fn ghost_screens(
    app: &mut AppState,
    overlays: &OverlayWindowManager<OpenXrOverlayData>,
    ghost: bool,
) {
    for o in overlays.values() {
        if o.config.backend.desktop_rect().is_none() || o.config.ghost == ghost {
            continue;
        }
        app.tasks.enqueue(TaskType::Overlay(OverlayTask::Modify(
            OverlaySelector::Name(o.config.name.clone()),
            Box::new(move |_app, owc| {
                owc.ghost = ghost;
                if let Some(state) = owc.active_state.as_mut() {
                    state.interactable = !ghost;
                }
            }),
        )));
    }
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
    // Where a region goes when it does not say: at its place on its screen's panel
    // (the panel's plane at its centre; a curved screen is treated as flat here).
    let default_pose = |r: &RegionSpec| -> Option<Vec<f32>> {
        overlays
            .values()
            .filter(|o| r.screen.as_deref().is_none_or(|n| &*o.config.name == n))
            .find_map(|o| {
                let d = o.config.backend.desktop_rect()?;
                let state = o.config.active_state.as_ref()?;
                if d[2] <= 0.0 || d[3] <= 0.0 {
                    return None;
                }
                let t = state.transform;
                let width = t.matrix3.col(0).length();
                let (_, rotation, _) = Mat4::from(t).to_scale_rotation_translation();
                let u = (r.rect[0] + r.rect[2] / 2.0 - d[0]) / d[2] - 0.5;
                let v = 0.5 - (r.rect[1] + r.rect[3] / 2.0 - d[1]) / d[3];
                let local = glam::Vec3::new(u * width, v * width * d[3] / d[2], 0.01);
                let pos = glam::Vec3::from(t.translation) + rotation * local;
                Some(
                    Mat4::from_rotation_translation(rotation, pos)
                        .to_cols_array()
                        .to_vec(),
                )
            })
    };
    ghost_screens(app, overlays, list.hide_screen && !list.regions.is_empty());
    app.region_popups = list
        .regions
        .iter()
        .filter(|r| r.popup)
        .map(|r| region::OpenPopup {
            parent: r.parent.clone(),
            rect: r.rect,
        })
        .collect();
    // A pop-up sits on its parent's panel, wherever that is now: at the offset it has from the
    // parent on the desktop, a little in front, at the parent's scale. Returns pose and width.
    let rects: HashMap<String, [f32; 4]> = list
        .regions
        .iter()
        .map(|r| (r.id.clone(), r.rect))
        .collect();
    let popup_pose = |r: &RegionSpec| -> Option<(Vec<f32>, f32)> {
        let parent = r.parent.as_ref()?;
        let prect = rects.get(parent)?;
        let name = region::overlay_name(parent);
        let o = overlays.values().find(|o| o.config.name == name)?;
        let t = o.config.active_state.as_ref()?.transform;
        let scale = t.matrix3.col(0).length();
        let aspect = (prect[2] / prect[3].max(1.0)).max(0.01);
        let parent_width = if aspect < 1.0 { scale * aspect } else { scale };
        let mpp = parent_width / prect[2].max(1.0);
        let (_, rotation, _) = Mat4::from(t).to_scale_rotation_translation();
        let dx = (r.rect[0] + r.rect[2] / 2.0) - (prect[0] + prect[2] / 2.0);
        let dy = (r.rect[1] + r.rect[3] / 2.0) - (prect[1] + prect[3] / 2.0);
        let pos = glam::Vec3::from(t.translation)
            + rotation * glam::Vec3::new(dx * mpp, -dy * mpp, 0.012);
        Some((
            Mat4::from_rotation_translation(rotation, pos)
                .to_cols_array()
                .to_vec(),
            r.rect[2] * mpp,
        ))
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
        for mut r in list.regions {
            let mut width_m = r.rect[2] * density(r.screen.as_deref());
            let is_new = !known.contains_key(&r.id);
            if r.popup
                && let Some((pose, width)) = popup_pose(&r)
            {
                r.pose = Some(pose);
                width_m = width;
            } else if is_new && r.pose.is_none() {
                r.pose = default_pose(&r);
            }
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
    if out.http.is_some() {
        if let Some(text) = REGION_INBOX.lock().ok().and_then(|mut i| i.take()) {
            if let Ok(mut seen) = REGION_SEEN.lock() {
                *seen = Some(std::time::Instant::now());
            }
            apply_regions(app, overlays, &text);
        } else if let Ok(mut seen) = REGION_SEEN.lock()
            && seen.is_some_and(|t| t.elapsed() > REGION_TIMEOUT)
        {
            *seen = None;
            log::info!("regions: the sender went quiet, removing its panels");
            apply_regions(app, overlays, r#"{"regions":[]}"#);
        }
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
