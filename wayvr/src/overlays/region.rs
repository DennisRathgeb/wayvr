//! Region panel: one rectangle of a mirrored screen as a panel of its own.
//!
//! A program that knows the layout of a window on the desktop (a browser extension that knows where
//! a page's toolbar or side panel is) sends rectangles; each becomes a panel that shows that part
//! of the screen's picture and sends the pointer to the matching spot of the desktop. The picture
//! is the screen panel's own (`AppState::screen_frames`), so nothing is captured twice; the screen
//! has to be shown for it to be current.

use std::sync::{Arc, Mutex};

use glam::{Affine2, Affine3A, DVec2, Quat, Vec3, vec2, vec3};
use serde::Deserialize;
use vulkano::{
    buffer::BufferUsage, image::view::ImageView, pipeline::graphics::color_blend::AttachmentBlend,
};
use wgui::gfx::{
    cmd::WGfxClearMode,
    pipeline::{WGfxPipeline, WPipelineCreateInfo},
};
use wlx_common::{
    overlays::{BackendAttrib, BackendAttribValue},
    windowing::{OverlayWindowState, Positioning},
};

use crate::{
    backend::input::{HoverResult, PointerHit, PointerMode},
    graphics::{ExtentExt, Vert2Uv, upload_quad_vertices},
    overlays::screen::backend::{can_move, set_next_move},
    state::AppState,
    subsystem::{
        hid::{MOUSE_LEFT, MOUSE_MIDDLE, MOUSE_RIGHT, WheelDelta},
        input::InputFocus,
    },
    windowing::{
        backend::{
            FrameMeta, OverlayBackend, OverlayEventData, RenderResources, ShouldRender,
            ui_transform,
        },
        window::{OverlayCategory, OverlayWindowConfig},
    },
};

/// The latest picture of a mirrored screen, kept by its panel for the region panels.
pub struct ScreenFrame {
    pub image: Arc<ImageView>,
    /// The desktop cursor on the screen, as fractions of its size.
    pub cursor: Option<(f32, f32)>,
    /// The screen's place and size on the desktop (logical pixels).
    pub logical: [f32; 4],
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct RegionSpec {
    pub id: String,
    /// The screen it is cut from; the first screen when absent.
    #[serde(default)]
    pub screen: Option<String>,
    /// x, y, width, height on the desktop, logical pixels.
    pub rect: [f32; 4],
    /// The visible shape inside the rectangle, as rectangles on the desktop (logical pixels):
    /// only these parts are drawn and take the pointer. The whole rectangle when empty.
    #[serde(default)]
    pub parts: Vec<[f32; 4]>,
    /// Where to put the panel: pose in the stage space, 4x4 in column order (rotation and
    /// translation; the size comes from the rectangle). Applied when it changes.
    #[serde(default)]
    pub pose: Option<Vec<f32>>,
}

pub type SharedSpec = Arc<Mutex<RegionSpec>>;

pub fn overlay_name(id: &str) -> Arc<str> {
    format!("region:{id}").into()
}

/// `width_m`: the panel's width in the room.
pub fn create_region_overlay(
    app: &mut AppState,
    spec: SharedSpec,
    width_m: f32,
) -> anyhow::Result<OverlayWindowConfig> {
    let key = std::env::var("WAYVR_BLACK_KEY").is_ok_and(|v| v != "0");
    let pipeline = app.gfx.create_pipeline(
        app.gfx_extras.shaders.get("vert_quad").unwrap(), // want panic
        app.gfx_extras
            .shaders
            .get(if key { "frag_screen_key" } else { "frag_screen" })
            .unwrap(), // want panic
        WPipelineCreateInfo::new(app.gfx.surface_format).use_blend(AttachmentBlend::default()),
    )?;
    let color_pipeline = app.gfx.create_pipeline(
        app.gfx_extras.shaders.get("vert_quad").unwrap(), // want panic
        app.gfx_extras.shaders.get("frag_color").unwrap(), // want panic
        WPipelineCreateInfo::new(app.gfx.surface_format).use_blend(AttachmentBlend::alpha()),
    )?;

    let (id, rect) = {
        let s = spec.lock().unwrap();
        (s.id.clone(), s.rect)
    };
    let aspect = (rect[2] / rect[3].max(1.0)).max(0.01);
    // The overlay's scale is its larger side.
    let scale = if aspect < 1.0 { width_m / aspect } else { width_m };

    Ok(OverlayWindowConfig {
        name: overlay_name(&id),
        default_state: OverlayWindowState {
            grabbable: true,
            interactable: true,
            positioning: Positioning::Floating,
            alpha: 1.0,
            transform: Affine3A::from_scale_rotation_translation(
                Vec3::ONE * scale,
                Quat::IDENTITY,
                vec3(0.0, 0.0, -0.7),
            ),
            ..OverlayWindowState::default()
        },
        input_focus: Some(InputFocus::PhysicalScreen),
        category: OverlayCategory::BuiltInPanel,
        show_on_spawn: true,
        ..OverlayWindowConfig::from_backend(Box::new(RegionBackend {
            spec,
            pipeline,
            color_pipeline,
            image: None,
            drawn: None,
            meta: None,
            mouse_transform: Affine2::IDENTITY,
            interaction_transform: None,
            uv: [0.0, 0.0, 1.0, 1.0],
            cursor: None,
            drawn_parts: 0,
        }))
    })
}

pub struct RegionBackend {
    spec: SharedSpec,
    pipeline: Arc<WGfxPipeline<Vert2Uv>>,
    color_pipeline: Arc<WGfxPipeline<Vert2Uv>>,
    image: Option<Arc<ImageView>>,
    /// What was drawn last: rectangle and cursor, to redraw when they change.
    drawn: Option<([f32; 4], Option<(f32, f32)>)>,
    meta: Option<FrameMeta>,
    mouse_transform: Affine2,
    interaction_transform: Option<Affine2>,
    /// The rectangle in the picture: u0, v0, u1, v1.
    uv: [f32; 4],
    /// The cursor inside this panel, in pixels of the panel.
    cursor: Option<(f32, f32)>,
    drawn_parts: usize,
}

impl RegionBackend {
    /// Whether the desktop point lies on the visible shape.
    fn on_shape(&self, pos: glam::Vec2) -> bool {
        let s = self.spec.lock().unwrap();
        s.parts.is_empty()
            || s.parts.iter().any(|p| {
                pos.x >= p[0] && pos.y >= p[1] && pos.x < p[0] + p[2] && pos.y < p[1] + p[3]
            })
    }
}

impl OverlayBackend for RegionBackend {
    fn init(&mut self, _app: &mut AppState) -> anyhow::Result<()> {
        Ok(())
    }
    fn pause(&mut self, _app: &mut AppState) -> anyhow::Result<()> {
        Ok(())
    }
    fn resume(&mut self, _app: &mut AppState) -> anyhow::Result<()> {
        self.drawn = None;
        Ok(())
    }

    fn should_render(&mut self, app: &mut AppState) -> anyhow::Result<ShouldRender> {
        let (rect, screen) = {
            let s = self.spec.lock().unwrap();
            (s.rect, s.screen.clone())
        };
        let frame = match screen {
            Some(name) => app.screen_frames.get(name.as_str()),
            None => app.screen_frames.values().next(),
        };
        let Some(frame) = frame else {
            return Ok(ShouldRender::Unable);
        };
        let [lx, ly, lw, lh] = frame.logical;
        if lw <= 0.0 || lh <= 0.0 || rect[2] < 1.0 || rect[3] < 1.0 {
            return Ok(ShouldRender::Unable);
        }
        let img = frame.image.extent_u32arr();
        let px = img[0] as f32 / lw; // picture pixels per logical pixel
        let extent = [
            ((rect[2] * px).round() as u32).max(2),
            ((rect[3] * px).round() as u32).max(2),
        ];
        self.uv = [
            (rect[0] - lx) / lw,
            (rect[1] - ly) / lh,
            (rect[0] - lx + rect[2]) / lw,
            (rect[1] - ly + rect[3]) / lh,
        ];
        self.cursor = frame.cursor.and_then(|(cx, cy)| {
            let (x, y) = (lx + cx * lw, ly + cy * lh);
            (x >= rect[0] && y >= rect[1] && x < rect[0] + rect[2] && y < rect[1] + rect[3])
                .then(|| ((x - rect[0]) * px, (y - rect[1]) * px))
        });
        self.mouse_transform = Affine2::from_cols(
            vec2(rect[2], 0.0),
            vec2(0.0, rect[3]),
            vec2(rect[0], rect[1]),
        );

        let new_image = self
            .image
            .as_ref()
            .is_none_or(|i| !Arc::ptr_eq(i, &frame.image));
        let new_extent = self.meta.is_none_or(|m| m.extent != extent);
        if new_extent {
            self.interaction_transform = Some(ui_transform(extent));
        }
        self.meta = Some(FrameMeta {
            extent,
            clear: WGfxClearMode::Clear([0.0, 0.0, 0.0, 0.0]),
            ..Default::default()
        });
        self.image = Some(frame.image.clone());

        let now = (rect, self.cursor);
        let parts_changed = self.spec.lock().unwrap().parts.len() != self.drawn_parts;
        if new_image || new_extent || parts_changed || self.drawn != Some(now) {
            Ok(ShouldRender::Should)
        } else {
            Ok(ShouldRender::Can)
        }
    }

    fn render(&mut self, app: &mut AppState, rdr: &mut RenderResources) -> anyhow::Result<()> {
        let Some(image) = self.image.clone() else {
            return Ok(());
        };
        let meta = self.meta.unwrap(); // want panic; set in should_render
        let extent = [meta.extent[0] as f32, meta.extent[1] as f32];
        let [u0, v0, u1, v1] = self.uv;

        // Each part of the shape as its own piece of the picture (the whole rectangle if no
        // shape was given): a..b are fractions of the panel.
        let (rect, parts) = {
            let s = self.spec.lock().unwrap();
            (s.rect, s.parts.clone())
        };
        let pieces: Vec<[f32; 4]> = if parts.is_empty() {
            vec![[0.0, 0.0, 1.0, 1.0]]
        } else {
            parts
                .iter()
                .map(|p| {
                    [
                        ((p[0] - rect[0]) / rect[2]).clamp(0.0, 1.0),
                        ((p[1] - rect[1]) / rect[3]).clamp(0.0, 1.0),
                        ((p[0] + p[2] - rect[0]) / rect[2]).clamp(0.0, 1.0),
                        ((p[1] + p[3] - rect[1]) / rect[3]).clamp(0.0, 1.0),
                    ]
                })
                .filter(|p| p[2] > p[0] && p[3] > p[1])
                .take(96)
                .collect()
        };
        for [a0, b0, a1, b1] in pieces {
            let (pu0, pu1) = (u0 + (u1 - u0) * a0, u0 + (u1 - u0) * a1);
            let (pv0, pv1) = (v0 + (v1 - v0) * b0, v0 + (v1 - v0) * b1);
            let buf = app
                .gfx
                .empty_buffer(BufferUsage::TRANSFER_DST | BufferUsage::VERTEX_BUFFER, 4)?;
            buf.write()?[0..4].copy_from_slice(&[
                Vert2Uv { in_pos: [a0, b0], in_uv: [pu0, pv0] },
                Vert2Uv { in_pos: [a0, b1], in_uv: [pu0, pv1] },
                Vert2Uv { in_pos: [a1, b0], in_uv: [pu1, pv0] },
                Vert2Uv { in_pos: [a1, b1], in_uv: [pu1, pv1] },
            ]);
            let set0 = self
                .pipeline
                .uniform_sampler(0, image.clone(), app.gfx.texture_filter)?;
            let pass = self.pipeline.create_pass(
                extent,
                [0.0, 0.0],
                buf,
                0..4,
                0..1,
                vec![set0],
                &Default::default(),
            )?;
            for cmd in &mut rdr.cmd_bufs {
                cmd.run_ref(&pass)?;
            }
        }

        // The desktop cursor: a small white square with a dark rim.
        if let Some((cx, cy)) = self.cursor {
            for (half, color) in [(8.0f32, [0.0f32, 0.0, 0.0, 1.0]), (5.0, [1.0, 1.0, 1.0, 1.0])] {
                let mut buf = app
                    .gfx
                    .empty_buffer(BufferUsage::TRANSFER_DST | BufferUsage::VERTEX_BUFFER, 4)?;
                upload_quad_vertices(
                    &mut buf,
                    extent[0],
                    extent[1],
                    cx - half,
                    cy - half,
                    half * 2.0,
                    half * 2.0,
                )?;
                let set0 = self.color_pipeline.uniform_buffer_upload(0, color.to_vec())?;
                let pass = self.color_pipeline.create_pass(
                    extent,
                    [0.0, 0.0],
                    buf,
                    0..4,
                    0..1,
                    vec![set0],
                    &Default::default(),
                )?;
                for cmd in &mut rdr.cmd_bufs {
                    cmd.run_ref(&pass)?;
                }
            }
        }

        self.drawn = Some((rect, self.cursor));
        self.drawn_parts = self.spec.lock().unwrap().parts.len();
        Ok(())
    }

    fn frame_meta(&mut self) -> Option<FrameMeta> {
        self.meta
    }

    fn notify(&mut self, _app: &mut AppState, _event_data: OverlayEventData) -> anyhow::Result<()> {
        Ok(())
    }

    fn on_hover(&mut self, app: &mut AppState, hit: &PointerHit) -> HoverResult {
        // outside the shape the pointer passes through to whatever is behind
        if !app.input_state.edit_mode && !self.on_shape(self.mouse_transform.transform_point2(hit.uv)) {
            return HoverResult::default();
        }
        let pick_now = app.input_state.picking_focus.is_picking();
        if pick_now {
            app.input_state.stop_picking();
            app.hid_provider
                .set_input_focus(app.wvr_server.as_mut(), InputFocus::PhysicalScreen);
        }
        if can_move()
            && !app.input_state.edit_mode
            && (!app.session.config.focus_follows_mouse_mode
                || app.input_state.pointers[hit.pointer].now.move_mouse)
        {
            let pos = self.mouse_transform.transform_point2(hit.uv);
            app.hid_provider
                .inner
                .mouse_move(DVec2::new(pos.x as _, pos.y as _));
            set_next_move(app.session.config.mouse_move_interval_ms as _);
        }
        HoverResult::consume()
    }

    fn on_pointer(&mut self, app: &mut AppState, hit: &PointerHit, pressed: bool) {
        let mut btn = match hit.mode {
            PointerMode::Right => MOUSE_RIGHT,
            PointerMode::Middle => MOUSE_MIDDLE,
            _ => MOUSE_LEFT,
        };
        if app.session.config.left_handed_mouse {
            btn = match btn {
                MOUSE_LEFT => MOUSE_RIGHT,
                MOUSE_RIGHT => MOUSE_LEFT,
                other => other,
            };
        }
        if pressed {
            set_next_move(app.session.config.click_freeze_time_ms as _);
        }
        app.hid_provider.inner.send_button(btn, pressed);
        if !pressed {
            return;
        }
        let pos = self.mouse_transform.transform_point2(hit.uv);
        app.hid_provider
            .inner
            .mouse_move(DVec2::new(pos.x as _, pos.y as _));
    }

    fn on_scroll(&mut self, app: &mut AppState, _hit: &PointerHit, mut delta: WheelDelta) {
        delta.x *= 60.0;
        delta.y *= 60.0;
        app.hid_provider.inner.wheel(delta);
    }

    fn on_left(&mut self, _app: &mut AppState, _hand: usize) {}

    fn get_interaction_transform(&mut self) -> Option<Affine2> {
        self.interaction_transform
    }
    fn get_attrib(&self, _attrib: BackendAttrib) -> Option<BackendAttribValue> {
        None
    }
    fn set_attrib(&mut self, _app: &mut AppState, _value: BackendAttribValue) -> bool {
        false
    }
}
