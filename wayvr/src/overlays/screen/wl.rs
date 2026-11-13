use glam::{DVec2, vec2};
use wgui::log::LogErr;
use wlx_capture::{
    WlxCapture,
    frame::Transform,
    wayland::{WlxClient, WlxOutput},
    wlr_screencopy::WlrScreencopyCapture,
};
use wlx_common::{astr_containers::AStrMapExt, config::CaptureMethod};

use crate::{
    overlays::screen::{backend::CaptureType, create_screen_from_backend, pw::ScreenCastBackend},
    state::{AppState, ScreenMeta},
    windowing::backend::OverlayBackend,
};

use super::{
    ScreenCreateData,
    backend::ScreenBackend,
    capture::{MainThreadWlxCapture, new_wlx_capture},
};

impl ScreenBackend {
    pub fn new_wlr_screencopy(output: &WlxOutput, app: &AppState) -> Option<Self> {
        let client = WlxClient::new()?;
        let capture = new_wlx_capture!(
            app.gfx_extras.queue_capture,
            WlrScreencopyCapture::new(client, output.id)
        );
        Some(Self::new_raw(
            output.name.clone(),
            app.feats.xr_backend,
            CaptureType::ScreenCopy,
            capture,
        ))
    }
}

pub fn create_screen_renderer_wl(
    output: &WlxOutput,
    has_wlr_screencopy: bool,
    kwin_nodes: Option<&std::collections::HashMap<String, u32>>,
    app: &mut AppState,
) -> anyhow::Result<Box<dyn OverlayBackend>> {
    if matches!(
        app.session.config.capture_method,
        CaptureMethod::ScreenCopyCpu | CaptureMethod::ScreenCopyGpu | CaptureMethod::Auto
    ) && has_wlr_screencopy
        && let Some(mut backend) = ScreenBackend::new_wlr_screencopy(output, app)
    {
        log::info!("{}: Using ScreenCopy capture", output.name);
        backend.logical_pos = vec2(output.logical_pos.0 as f32, output.logical_pos.1 as f32);
        backend.logical_size = vec2(output.logical_size.0 as f32, output.logical_size.1 as f32);
        backend.apply_mouse_transform_with_override(Transform::Undefined);
        return Ok(Box::new(backend));
    }

    // KWin hands out the stream of exactly this output: no dialog, no wrong pick.
    if let Some(node_id) = kwin_nodes.and_then(|n| n.get(&*output.name)) {
        use wlx_capture::pipewire::capture::PipewireCapture;

        log::info!("{}: Using the stream from KWin", output.name);
        let capture = new_wlx_capture!(
            app.gfx_extras.queue_capture,
            PipewireCapture::new(output.name.clone(), *node_id)
        );
        let mut backend = ScreenBackend::new_raw(
            output.name.clone(),
            app.feats.xr_backend,
            CaptureType::PipeWire,
            capture,
        );
        backend.logical_pos = vec2(output.logical_pos.0 as f32, output.logical_pos.1 as f32);
        backend.logical_size = vec2(output.logical_size.0 as f32, output.logical_size.1 as f32);
        backend.apply_mouse_transform_with_override(Transform::Undefined);
        return Ok(Box::new(backend));
    }

    log::info!("{}: Using Pipewire capture", output.name);
    let display_name = &*output.name;

    // Find existing token by display
    let token = app
        .session
        .pw_tokens
        .arc_get(display_name)
        .map(|x| x.clone().into());

    if token.is_some() {
        log::info!("Found existing Pipewire token for display {display_name}");
    }

    Ok(Box::new(
        ScreenCastBackend::new_wl(output, token, app)
            .log_err("Failed to create screen with screen cast backend")?,
    ))
}

pub fn create_screens_wayland(
    wl: &mut WlxClient,
    app: &mut AppState,
) -> anyhow::Result<ScreenCreateData> {
    let mut screens = vec![];
    let mut order = vec![];
    let has_wlr_screencopy = wl.maybe_wlr_screencopy_mgr.is_some();

    // Every screen straight from KWin where it lets us; the portal with its dialog otherwise.
    let kwin_nodes = if has_wlr_screencopy {
        None
    } else {
        wlx_capture::kde_screencast::stream_outputs(true)
    };

    for (id, output) in &wl.outputs {
        if app.screens.iter().any(|s| s.name == output.name) {
            continue;
        }

        log::info!(
            "{}: Init screen of res {:?}, logical {:?} at {:?}",
            output.name,
            output.size,
            output.logical_size,
            output.logical_pos,
        );

        let backend =
            create_screen_renderer_wl(output, has_wlr_screencopy, kwin_nodes.as_ref(), app)?;
        let window_config = create_screen_from_backend(
            output.name.clone(),
            output.transform,
            &app.session,
            backend,
        );

        let meta = ScreenMeta {
            name: wl.outputs[id].name.clone(),
            native_handle: *id,
        };

        order.push(output.logical_pos.0);
        screens.push((meta, window_config));
    }

    // Side by side on an arc around the viewer, left to right as on the desktop.
    {
        const RADIUS: f32 = 1.6;
        let mut by_x: Vec<usize> = (0..screens.len()).collect();
        by_x.sort_by_key(|i| order[*i]);
        let mid = (screens.len() as f32 - 1.0) / 2.0;
        for (slot, i) in by_x.into_iter().enumerate() {
            let tf = &mut screens[i].1.default_state.transform;
            let width = tf.x_axis.length();
            let angle = (slot as f32 - mid) * (width * 1.03 / RADIUS);
            let pivot = tf.translation + glam::Vec3A::Z * RADIUS;
            tf.matrix3 = glam::Mat3A::from_rotation_y(-angle) * tf.matrix3;
            tf.translation = pivot + glam::Vec3A::new(angle.sin(), 0.0, -angle.cos()) * RADIUS;
        }
    }

    let extent = wl.get_desktop_extent();
    let origin = wl.get_desktop_origin();

    app.hid_provider
        .inner
        .set_desktop_extent(DVec2::new(extent.0 as _, extent.1 as _));
    app.hid_provider
        .inner
        .set_desktop_origin(DVec2::new(origin.0 as _, origin.1 as _));

    Ok(ScreenCreateData { screens })
}
