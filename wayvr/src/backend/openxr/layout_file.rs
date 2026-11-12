//! Publishes where the panels are in the room, for programs that place their own content relative
//! to a screen: `$XDG_RUNTIME_DIR/wayvr-layout.json`, rewritten about ten times a second while
//! `WAYVR_LAYOUT_FILE` is set. Poses are in the stage space, as 4x4 matrices in column order.

use std::{io::Write, path::PathBuf, sync::OnceLock};

use glam::{Affine3A, Mat4};
use serde_json::json;

use super::overlay::OpenXrOverlayData;
use crate::{state::AppState, windowing::manager::OverlayWindowManager};

fn path() -> Option<&'static PathBuf> {
    static PATH: OnceLock<Option<PathBuf>> = OnceLock::new();
    PATH.get_or_init(|| {
        if std::env::var("WAYVR_LAYOUT_FILE").is_ok_and(|v| v != "0") {
            std::env::var("XDG_RUNTIME_DIR")
                .ok()
                .map(|d| PathBuf::from(d).join("wayvr-layout.json"))
        } else {
            None
        }
    })
    .as_ref()
}

fn mat(a: &Affine3A) -> Vec<f32> {
    Mat4::from(*a).to_cols_array().to_vec()
}

pub(super) fn tick(
    app: &AppState,
    overlays: &OverlayWindowManager<OpenXrOverlayData>,
    frame: usize,
) {
    let Some(path) = path() else {
        return;
    };
    if frame % 6 != 0 {
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
            }))
        })
        .collect();
    let doc = json!({
        "frame": frame,
        "hmd": mat(&app.input_state.hmd),
        "overlays": list,
    });
    let tmp = path.with_extension("json.tmp");
    let ok = std::fs::File::create(&tmp)
        .and_then(|mut f| f.write_all(doc.to_string().as_bytes()))
        .and_then(|()| std::fs::rename(&tmp, path));
    if let Err(e) = ok {
        log::debug!("layout file: {e}");
    }
}
