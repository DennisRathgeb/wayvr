//! Publishes where the panels are in the room, for programs that place their own content relative
//! to a screen (a browser extension that puts a 3D model where a page shows it flat). Poses are in
//! the stage space, as 4x4 matrices in column order; mirrored screens carry their place on the
//! desktop. Updated about ten times a second, on request:
//!
//! - `WAYVR_LAYOUT_PORT=<port>`: served as JSON over HTTP on 127.0.0.1:<port> (any path).
//! - `WAYVR_LAYOUT_FILE=1`: written to `$XDG_RUNTIME_DIR/wayvr-layout.json`.

use std::{
    io::{Read, Write},
    net::TcpListener,
    path::PathBuf,
    sync::{Arc, Mutex, OnceLock},
    time::Duration,
};

use glam::{Affine3A, Mat4};
use serde_json::json;

use super::overlay::OpenXrOverlayData;
use crate::{state::AppState, windowing::manager::OverlayWindowManager};

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
            let mut buf = [0u8; 2048];
            let _ = stream.read(&mut buf);
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
    app: &AppState,
    overlays: &OverlayWindowManager<OpenXrOverlayData>,
    frame: usize,
) {
    let out = outputs();
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
