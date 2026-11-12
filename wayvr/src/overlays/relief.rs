//! Relief panel: shows the 3D view of a page running in a WayVR window as a surface in space.
//!
//! The page publishes the depth of its 3D view as a strip of pixels at the bottom left of its
//! window (onshape-xr, `extension-depth/depth.js`). This panel samples the window's picture
//! (colour and depth from the same frame) and renders, for each eye, what that eye sees of the
//! surface through the panel. The panel is an ordinary overlay: it is placed, grabbed and scaled
//! like any other.

use std::sync::Arc;

use glam::{Affine2, Affine3A, Quat, Vec3, Vec3A, vec3};
use smithay::wayland::compositor::with_states;
use vulkano::{image::view::ImageView, pipeline::graphics::color_blend::AttachmentBlend};
use wgui::gfx::{
    cmd::WGfxClearMode,
    pipeline::{WGfxPipeline, WPipelineCreateInfo},
};
use wlx_common::{
    overlays::{BackendAttrib, BackendAttribValue, StereoMode},
    windowing::{OverlayWindowState, Positioning},
};

use crate::{
    backend::{
        input::{HoverResult, PointerHit},
        task::{OverlayTask, TaskType},
        wayvr::{SurfaceBufWithImage, window::WindowHandle},
    },
    graphics::Vert2Uv,
    state::AppState,
    subsystem::hid::WheelDelta,
    windowing::{
        OverlayID, OverlaySelector,
        backend::{FrameMeta, OverlayBackend, OverlayEventData, RenderResources, ShouldRender},
        window::{OverlayCategory, OverlayWindowConfig},
    },
};

/// Size of the per-eye picture of the panel.
const EXTENT: [u32; 2] = [1280, 800];
/// Width of the panel in metres when it is spawned.
const WIDTH: f32 = 0.6;
/// Size of the view's box relative to the panel (room around it for parallax).
const BOX: f32 = 0.7;

pub fn enabled() -> bool {
    std::env::var("WAYVR_RELIEF").is_ok_and(|v| v != "0")
}

pub fn create_relief_overlay(
    name: Arc<str>,
    app: &mut AppState,
    window: WindowHandle,
    icon: Arc<str>,
) -> anyhow::Result<OverlayWindowConfig> {
    let pipeline = app.gfx.create_pipeline(
        app.gfx_extras.shaders.get("vert_relief").unwrap(), // want panic
        app.gfx_extras.shaders.get("frag_relief").unwrap(), // want panic
        WPipelineCreateInfo::new(app.gfx.surface_format).use_blend(AttachmentBlend::alpha()),
    )?;

    Ok(OverlayWindowConfig {
        name: format!("{name} 3D").into(),
        default_state: OverlayWindowState {
            grabbable: true,
            interactable: false,
            positioning: Positioning::Floating,
            alpha: 1.0,
            transform: Affine3A::from_scale_rotation_translation(
                Vec3::ONE * WIDTH,
                Quat::IDENTITY,
                vec3(0.0, -0.15, -0.8),
            ),
            ..OverlayWindowState::default()
        },
        category: OverlayCategory::WayVR,
        show_on_spawn: true,
        ..OverlayWindowConfig::from_backend(Box::new(ReliefBackend {
            window,
            icon,
            pipeline,
            image: None,
            transform: Affine3A::IDENTITY,
            id: None,
            editing: false,
        }))
    })
}

pub struct ReliefBackend {
    window: WindowHandle,
    icon: Arc<str>,
    pipeline: Arc<WGfxPipeline<Vert2Uv>>,
    image: Option<Arc<ImageView>>,
    transform: Affine3A,
    id: Option<OverlayID>,
    editing: bool,
}

impl OverlayBackend for ReliefBackend {
    fn init(&mut self, _app: &mut AppState) -> anyhow::Result<()> {
        Ok(())
    }
    fn pause(&mut self, _app: &mut AppState) -> anyhow::Result<()> {
        Ok(())
    }
    fn resume(&mut self, _app: &mut AppState) -> anyhow::Result<()> {
        Ok(())
    }

    fn should_render(&mut self, app: &mut AppState) -> anyhow::Result<ShouldRender> {
        let Some(toplevel) = app
            .wvr_server
            .as_ref()
            .and_then(|sv| sv.wm.windows.get(self.window))
            .map(|win| win.toplevel.clone())
        else {
            // The window is gone: so is its 3D view.
            if let Some(id) = self.id.take() {
                app.tasks
                    .enqueue(TaskType::Overlay(OverlayTask::Drop(OverlaySelector::Id(id))));
            }
            return Ok(ShouldRender::Unable);
        };

        let Some(surf) = with_states(toplevel.wl_surface(), SurfaceBufWithImage::get_from_surface)
        else {
            return Ok(ShouldRender::Unable);
        };
        self.image = Some(surf.image);

        // The picture depends on where the eyes are, so it is drawn every frame.
        Ok(ShouldRender::Should)
    }

    fn render(&mut self, app: &mut AppState, rdr: &mut RenderResources) -> anyhow::Result<()> {
        let Some(image) = self.image.clone() else {
            return Ok(());
        };
        let to_panel = self.transform.inverse();
        let aspect = EXTENT[0] as f32 / EXTENT[1] as f32;
        let extent = [EXTENT[0] as f32, EXTENT[1] as f32];

        for (i, buf) in rdr.cmd_bufs.iter_mut().enumerate() {
            let eye: Vec3A = to_panel.transform_point3a(app.input_state.eyes[i.min(1)]);
            let set0 = self.pipeline.uniform_sampler(
                0,
                image.clone(),
                vulkano::image::sampler::Filter::Linear,
            )?;
            let set1 = self.pipeline.uniform_buffer_upload(
                1,
                vec![
                    eye.x,
                    eye.y,
                    eye.z,
                    0.0,
                    BOX,
                    aspect,
                    if self.editing { 1.0 } else { 0.0 },
                    0.0,
                ],
            )?;
            let pass = self.pipeline.create_pass(
                extent,
                [0.0, 0.0],
                app.gfx_extras.quad_verts.clone(),
                0..4,
                0..1,
                vec![set0, set1],
                &Default::default(),
            )?;
            buf.run_ref(&pass)?;
        }
        Ok(())
    }

    fn frame_meta(&mut self) -> Option<FrameMeta> {
        Some(FrameMeta {
            extent: EXTENT,
            clear: WGfxClearMode::Clear([0.0, 0.0, 0.0, 0.0]),
            stereo: StereoMode::LeftRight,
            ..Default::default()
        })
    }

    fn set_world_transform(&mut self, transform: Affine3A) {
        self.transform = transform;
    }

    fn notify(&mut self, _app: &mut AppState, event_data: OverlayEventData) -> anyhow::Result<()> {
        match event_data {
            OverlayEventData::IdAssigned(id) => self.id = Some(id),
            OverlayEventData::EditModeChanged(editing) => self.editing = editing,
            _ => {}
        }
        Ok(())
    }

    fn on_hover(&mut self, _app: &mut AppState, _hit: &PointerHit) -> HoverResult {
        HoverResult::default()
    }
    fn on_left(&mut self, _app: &mut AppState, _pointer: usize) {}
    fn on_pointer(&mut self, _app: &mut AppState, _hit: &PointerHit, _pressed: bool) {}
    fn on_scroll(&mut self, _app: &mut AppState, _hit: &PointerHit, _delta: WheelDelta) {}
    fn get_interaction_transform(&mut self) -> Option<Affine2> {
        None
    }
    fn get_attrib(&self, attrib: BackendAttrib) -> Option<BackendAttribValue> {
        match attrib {
            BackendAttrib::Icon => Some(BackendAttribValue::Icon(self.icon.clone())),
            _ => None,
        }
    }
    fn set_attrib(&mut self, _app: &mut AppState, _value: BackendAttribValue) -> bool {
        false
    }
}
