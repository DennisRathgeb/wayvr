//! Calibration board: the printed ChArUco board drawn in space where the tracking camera saw it,
//! so the wearer can judge the display's geometry (eye position, field of view) by how well the
//! drawn board covers the real one from several places.
//!
//! The pose comes from `$XDG_RUNTIME_DIR/xr-calib-board.json`, written by `calib/board_publisher.py`:
//! `{"position": [x, y, z], "orientation": [x, y, z, w], "width": 0.75, "height": 0.525,
//! "squares": [10, 7], "margin": 0.03}` in the stage space, the panel's centre and its facing
//! (the panel's -Z looks at the wearer). The file is read again whenever it changes; the board is
//! shown while the file is younger than two seconds.
use std::{path::PathBuf, sync::Arc, time::SystemTime};

use glam::{Affine2, Affine3A, Quat, Vec3};
use vulkano::pipeline::graphics::color_blend::AttachmentBlend;
use wgui::gfx::{
    cmd::WGfxClearMode,
    pipeline::{WGfxPipeline, WPipelineCreateInfo},
};
use wlx_common::{
    overlays::{BackendAttrib, BackendAttribValue},
    windowing::{OverlayWindowState, Positioning},
};

use crate::{
    backend::input::{HoverResult, PointerHit},
    graphics::Vert2Uv,
    state::AppState,
    subsystem::hid::WheelDelta,
    windowing::{
        backend::{FrameMeta, OverlayBackend, OverlayEventData, RenderResources, ShouldRender},
        window::{OverlayCategory, OverlayWindowConfig},
    },
};

pub const CALIB_BOARD_NAME: &str = "calib-board";
const EXTENT: [u32; 2] = [1500, 1050];

pub fn pose_file() -> PathBuf {
    let dir = std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".into());
    PathBuf::from(dir).join("xr-calib-board.json")
}

#[derive(Default, Clone, Copy)]
struct BoardPose {
    position: Vec3,
    orientation: Quat,
    width: f32,
    height: f32,
    squares: [f32; 2],
    margin: f32,
}

fn read_pose(path: &PathBuf) -> Option<BoardPose> {
    let text = std::fs::read_to_string(path).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    let arr = |key: &str, n: usize| -> Option<Vec<f32>> {
        let a = v.get(key)?.as_array()?;
        if a.len() != n {
            return None;
        }
        a.iter().map(|x| x.as_f64().map(|f| f as f32)).collect()
    };
    let p = arr("position", 3)?;
    let q = arr("orientation", 4)?;
    let s = arr("squares", 2).unwrap_or_else(|| vec![10.0, 7.0]);
    Some(BoardPose {
        position: Vec3::new(p[0], p[1], p[2]),
        orientation: Quat::from_xyzw(q[0], q[1], q[2], q[3]).normalize(),
        width: v.get("width").and_then(|x| x.as_f64()).unwrap_or(0.75) as f32,
        height: v.get("height").and_then(|x| x.as_f64()).unwrap_or(0.525) as f32,
        squares: [s[0], s[1]],
        margin: v.get("margin").and_then(|x| x.as_f64()).unwrap_or(0.03) as f32,
    })
}

pub fn create_calib_board(app: &mut AppState) -> anyhow::Result<OverlayWindowConfig> {
    let pipeline = app.gfx.create_pipeline(
        app.gfx_extras.shaders.get("vert_quad").unwrap(), // want panic
        app.gfx_extras.shaders.get("frag_checker").unwrap(), // want panic
        WPipelineCreateInfo::new(app.gfx.surface_format).use_blend(AttachmentBlend::alpha()),
    )?;
    Ok(OverlayWindowConfig {
        name: CALIB_BOARD_NAME.into(),
        default_state: OverlayWindowState {
            grabbable: false,
            interactable: false,
            positioning: Positioning::Static,
            alpha: 1.0,
            transform: Affine3A::from_scale_rotation_translation(
                Vec3::ONE * 0.75,
                Quat::IDENTITY,
                Vec3::new(0.0, 0.0, -1.0),
            ),
            ..OverlayWindowState::default()
        },
        category: OverlayCategory::Internal,
        show_on_spawn: false,
        global: true,
        ..OverlayWindowConfig::from_backend(Box::new(CalibBoardBackend {
            pipeline,
            path: pose_file(),
            mtime: None,
            pose: None,
            last_check: std::time::Instant::now(),
            dirty: true,
        }))
    })
}

pub struct CalibBoardBackend {
    pipeline: Arc<WGfxPipeline<Vert2Uv>>,
    path: PathBuf,
    mtime: Option<SystemTime>,
    pose: Option<BoardPose>,
    last_check: std::time::Instant,
    dirty: bool,
}

impl CalibBoardBackend {
    /// The drawing parameters (squares, margin) follow the pose file; re-read when it changes.
    fn poll_params(&mut self) {
        if self.last_check.elapsed().as_millis() < 500 {
            return;
        }
        self.last_check = std::time::Instant::now();
        let mtime = std::fs::metadata(&self.path).ok().and_then(|m| m.modified().ok());
        if mtime == self.mtime {
            return;
        }
        self.mtime = mtime;
        if let Some(p) = read_pose(&self.path) {
            let changed = self.pose.is_none_or(|o| o.squares != p.squares || o.margin != p.margin);
            self.pose = Some(p);
            if changed {
                self.dirty = true;
            }
        }
    }
}

static POLL: std::sync::Mutex<Option<(Option<SystemTime>, std::time::Instant, bool)>> = std::sync::Mutex::new(None);

/// Called from the frame loop: places or hides the board overlay as the pose file changes
/// (looked at five times a second).
pub fn update<T>(app: &mut AppState, overlays: &mut crate::windowing::manager::OverlayWindowManager<T>)
where
    T: Default,
{
    let path = pose_file();
    let change: Option<Option<BoardPose>> = {
        let mut guard = POLL.lock().unwrap();
        let (last_mtime, last_check, shown) = guard.get_or_insert((None, std::time::Instant::now() - std::time::Duration::from_secs(1), false));
        if last_check.elapsed().as_millis() < 200 {
            None
        } else {
            *last_check = std::time::Instant::now();
            let mtime = std::fs::metadata(&path).ok().and_then(|m| m.modified().ok());
            let fresh = mtime.and_then(|t| t.elapsed().ok()).is_some_and(|age| age.as_secs_f32() < 2.0);
            if !fresh {
                if *shown {
                    *shown = false;
                    *last_mtime = None;
                    Some(None)
                } else {
                    None
                }
            } else if mtime == *last_mtime {
                None
            } else {
                *last_mtime = mtime;
                match read_pose(&path) {
                    Some(p) => {
                        *shown = true;
                        Some(Some(p))
                    }
                    None => None,
                }
            }
        }
    };
    let Some(change) = change else {
        return;
    };
    let Some((_, overlay)) = overlays.iter_mut().find(|(_, o)| &*o.config.name == CALIB_BOARD_NAME) else {
        return;
    };
    match change {
        Some(p) => {
            let transform = Affine3A::from_scale_rotation_translation(Vec3::splat(p.width), p.orientation, p.position);
            overlay.config.default_state.transform = transform;
            if overlay.config.active_state.is_none() {
                overlay.config.activate(app, false);
            }
            if let Some(state) = overlay.config.active_state.as_mut() {
                state.transform = transform;
                state.positioning = Positioning::Static;
            }
            overlay.config.dirty = true;
        }
        None => overlay.config.deactivate(),
    }
}

impl OverlayBackend for CalibBoardBackend {
    fn init(&mut self, _app: &mut AppState) -> anyhow::Result<()> {
        Ok(())
    }
    fn pause(&mut self, _app: &mut AppState) -> anyhow::Result<()> {
        Ok(())
    }
    fn resume(&mut self, _app: &mut AppState) -> anyhow::Result<()> {
        Ok(())
    }
    fn should_render(&mut self, _app: &mut AppState) -> anyhow::Result<ShouldRender> {
        self.poll_params();
        if self.dirty {
            self.dirty = false;
            return Ok(ShouldRender::Should);
        }
        Ok(ShouldRender::Can)
    }
    fn render(&mut self, app: &mut AppState, rdr: &mut RenderResources) -> anyhow::Result<()> {
        let p = self.pose.unwrap_or(BoardPose {
            squares: [10.0, 7.0],
            margin: 0.03,
            width: 0.75,
            height: 0.525,
            ..Default::default()
        });
        let extent = [EXTENT[0] as f32, EXTENT[1] as f32];
        for buf in rdr.cmd_bufs.iter_mut() {
            let set0 = self.pipeline.uniform_buffer_upload(
                0,
                vec![p.squares[0], p.squares[1], p.margin, 0.0, 0.1, 0.1, 0.1, 0.85],
            )?;
            let pass = self.pipeline.create_pass(
                extent,
                [0.0, 0.0],
                app.gfx_extras.quad_verts.clone(),
                0..4,
                0..1,
                vec![set0],
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
            ..Default::default()
        })
    }
    fn notify(&mut self, _app: &mut AppState, _event_data: OverlayEventData) -> anyhow::Result<()> {
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
    fn get_attrib(&self, _attrib: BackendAttrib) -> Option<BackendAttribValue> {
        None
    }
    fn set_attrib(&mut self, _app: &mut AppState, _value: BackendAttribValue) -> bool {
        false
    }
}
