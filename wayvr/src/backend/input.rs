use std::f32::consts::PI;
use std::process::{Child, Command};
use std::sync::Arc;
use std::time::{Duration, Instant};

use glam::{Affine3A, Quat, Vec2, Vec3A, Vec3Swizzles};

use idmap_derive::IntegerId;
use smallvec::{SmallVec, smallvec};
use strum::{AsRefStr, EnumIs};
use wayvr_ipc::packet_client::{HandsfreeAction, HandsfreeParams};
use wlx_common::common::LeftRight;
use wlx_common::windowing::{OverlayWindowState, Positioning};

use crate::backend::task::{InputTask, OverlayTask};
use crate::overlays::anchor::{ANCHOR_NAME, GRAB_HELP_NAME};
use crate::overlays::watch::WATCH_NAME;
use crate::state::{AppSession, AppState};
use crate::subsystem::hid::WheelDelta;
use crate::subsystem::input::InputFocus;
use crate::windowing::backend::OverlayEventData;
use crate::windowing::manager::OverlayWindowManager;
use crate::windowing::window::{
    self, OverlayCategory, OverlayWindowData, realign, realign_roll, scalar_scale,
};
use crate::windowing::{OverlayID, OverlaySelector};

use super::task::TaskType;

#[derive(Clone, Default)]
pub struct HoverResult {
    pub haptics: Option<Haptics>,
    /// If true, the laser shows at this position and no further raycasting will be done.
    pub consume: bool,
}

impl HoverResult {
    pub fn consume() -> Self {
        HoverResult {
            consume: true,
            ..Default::default()
        }
    }
}

pub struct TrackedDevice {
    pub soc: Option<f32>,
    pub charging: bool,
    pub role: TrackedDeviceRole,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, IntegerId, AsRefStr)]
pub enum TrackedDeviceRole {
    None,
    Hmd,
    LeftHand,
    RightHand,
    Tracker,
}

#[derive(Debug, Eq, PartialEq, Clone, Copy, EnumIs)]
pub enum FocusPickState {
    None,
    /// aiming to pick the focus
    Aiming,
    /// picking the focus. this will be consumed on valid on_hover
    Picking,
}

/// The mouse pointer: a ray from the head whose direction is fixed in the world, not the head,
/// and is turned by the captured mouse. Yaw about +Y, pitch about the ray's X, in radians.
#[derive(Debug, Clone, Copy, Default)]
pub struct MouseRay {
    pub yaw: f32,
    pub pitch: f32,
    /// Wheel notches since the last frame (positive = wheel up = nearer); consumed once a frame
    /// to push or pull the panel under the pointer.
    pub push: f32,
    /// The same for all screens at once (wheel with Meta+Alt held).
    pub push_all: f32,
    /// Steps to make all screens larger (positive) or smaller (Meta+Alt+Up / Down).
    pub scale_all: f32,
}

impl MouseRay {
    const MAX_PITCH: f32 = 85.0 * PI / 180.0;

    /// Turn by a mouse motion (right and down are positive), `sensitivity` in radians per unit.
    pub fn turn(&mut self, dx: f32, dy: f32, sensitivity: f32) {
        self.yaw = (self.yaw - dx * sensitivity).rem_euclid(2.0 * PI);
        self.pitch = (self.pitch - dy * sensitivity).clamp(-Self::MAX_PITCH, Self::MAX_PITCH);
    }

    /// Point where the head looks.
    pub fn aim_like(&mut self, hmd: &Affine3A) {
        let fwd = hmd.transform_vector3a(Vec3A::NEG_Z).normalize_or_zero();
        self.yaw = (-fwd.x).atan2(-fwd.z);
        self.pitch = fwd.y.clamp(-1.0, 1.0).asin();
    }

    pub fn rotation(&self) -> Quat {
        Quat::from_rotation_y(self.yaw) * Quat::from_rotation_x(self.pitch)
    }
}

pub struct InputState {
    pub hmd: Affine3A,
    /// Left and right eye in the stage space.
    pub eyes: [glam::Vec3A; 2],
    pub ipd: f32,
    pub pointers: [Pointer; 2],
    pub devices: Vec<TrackedDevice>,
    pub handsfree_state: PointerState,
    pub mouse_ray: MouseRay,
    /// While Meta+Alt is held: since when, and (once it has started) the anchor relative to the
    /// head. All anchored panels are carried with the head as one rigid group.
    pub carry: Option<(Instant, Option<Affine3A>)>,
    /// The radius of the circle of screens around the head, once it has been set with
    /// Meta+Alt+wheel: a recentre lays the screens out at this radius again.
    pub screen_radius: Option<f32>,
    /// Panels held by a pinch, per hand (hand_pinch_grab).
    pub pinch_grabs: [Option<PinchGrab>; 2],
    /// Frames in a row with the pinch closed, per hand, while nothing is held.
    pub pinch_closed: [u32; 2],
    /// Mirrored from the overlay manager every frame, so input routing can see it.
    pub edit_mode: bool,
    pub picking_focus: FocusPickState,
    processes: Vec<Child>,
    disable_lerp_until: Instant,
}

impl InputState {
    pub fn new() -> Self {
        Self {
            hmd: Affine3A::IDENTITY,
            eyes: [glam::Vec3A::ZERO; 2],
            ipd: 0.0,
            pointers: [Pointer::new(0), Pointer::new(1)],
            devices: Vec::new(),
            processes: Vec::new(),
            handsfree_state: PointerState::default(),
            mouse_ray: MouseRay::default(),
            carry: None,
            screen_radius: None,
            pinch_grabs: [None, None],
            pinch_closed: [0, 0],
            edit_mode: false,
            picking_focus: FocusPickState::None,
            disable_lerp_until: Instant::now(),
        }
    }

    pub fn should_disable_lerp(&self) -> bool {
        self.disable_lerp_until > Instant::now()
    }

    pub fn stop_picking(&mut self) {
        if !self.picking_focus.is_none() {
            self.picking_focus = FocusPickState::None;
            self.disable_lerp_until = Instant::now() + Duration::from_millis(250);
        }
    }

    pub fn apply_handsfree_action(&mut self, params: HandsfreeParams) {
        const fn set_true(v: &mut bool) {
            *v = true;
        }
        const fn set_false(v: &mut bool) {
            *v = false;
        }
        const fn toggle(v: &mut bool) {
            *v = !*v;
        }

        let (action, apply) = match params {
            HandsfreeParams::SetMode(_) => {
                return;
            }
            HandsfreeParams::Press(action) => (action, set_true as fn(&mut bool)),
            HandsfreeParams::Release(action) => (action, set_false as fn(&mut bool)),
            HandsfreeParams::Toggle(action) => (action, toggle as fn(&mut bool)),
            HandsfreeParams::Scroll(val) => {
                self.handsfree_state.scroll_y = val;
                return;
            }
        };

        match action {
            HandsfreeAction::Click => apply(&mut self.handsfree_state.click),
            HandsfreeAction::RightModifier => apply(&mut self.handsfree_state.click_modifier_right),
            HandsfreeAction::MiddleModifier => {
                apply(&mut self.handsfree_state.click_modifier_middle);
            }
            HandsfreeAction::Grab => apply(&mut self.handsfree_state.grab),
        }
    }

    pub fn handle_task(&mut self, task: InputTask) {
        match task {
            InputTask::Haptics { device, haptics } => {
                if let Some(pointer) = self.pointers.get_mut(device) {
                    pointer.pending_haptics = Some(haptics);
                } else {
                    log::warn!("Can't trigger haptics on non-existing device: {device}");
                }
            }
        }
    }

    pub const fn pre_update(&mut self) {
        self.pointers[0].before = self.pointers[0].now;
        self.pointers[1].before = self.pointers[1].now;
    }

    pub fn post_update(&mut self, session: &AppSession) {
        for hand in &mut self.pointers {
            #[cfg(debug_assertions)]
            debug_print_hand(hand);

            if hand.now.click {
                hand.last_click = Instant::now();
            }

            // Prevent the mode from changing during a click
            if !hand.before.click {
                if hand.now.click_modifier_right {
                    hand.interaction.mode = PointerMode::Right;
                    continue;
                }

                if hand.now.click_modifier_middle {
                    hand.interaction.mode = PointerMode::Middle;
                    continue;
                }

                let hmd_up = self.hmd.transform_vector3a(Vec3A::Y);
                let dot = hmd_up.dot(hand.pose.transform_vector3a(Vec3A::X))
                    * 2.0f32.mul_add(-(hand.idx as f32), 1.0);

                hand.interaction.mode = if dot < -0.85 {
                    PointerMode::Right
                } else if dot > 0.7 {
                    PointerMode::Middle
                } else {
                    PointerMode::Left
                };

                let middle_click_orientation = false;
                let right_click_orientation = false;
                match hand.interaction.mode {
                    PointerMode::Middle => {
                        if !middle_click_orientation {
                            hand.interaction.mode = PointerMode::Left;
                        }
                    }
                    PointerMode::Right if !right_click_orientation => {
                        hand.interaction.mode = PointerMode::Left;
                    }
                    _ => {}
                }
            }

            if hand.now.alt_click != hand.before.alt_click {
                // Reap previous processes
                self.processes
                    .retain_mut(|child| !matches!(child.try_wait(), Ok(Some(_))));

                let mut args = if hand.now.alt_click {
                    session.config.alt_click_down.iter()
                } else {
                    session.config.alt_click_up.iter()
                };

                if let Some(program) = args.next()
                    && let Ok(child) = Command::new(program).args(args).spawn()
                {
                    self.processes.push(child);
                }
            }
        }
    }
}

#[cfg(debug_assertions)]
fn debug_print_hand(hand: &Pointer) {
    {
        if hand.now.click != hand.before.click {
            log::debug!("Hand {}: click {}", hand.idx, hand.now.click);
        }
        if hand.now.grab != hand.before.grab {
            log::debug!("Hand {}: grab {}", hand.idx, hand.now.grab);
        }
        if hand.now.grab_float != hand.before.grab_float {
            log::debug!("Hand {}: grab {}", hand.idx, hand.now.grab);
        }
        if hand.now.alt_click != hand.before.alt_click {
            log::debug!("Hand {}: alt_click {}", hand.idx, hand.now.alt_click);
        }
        if hand.now.show_hide != hand.before.show_hide {
            log::debug!("Hand {}: show_hide {}", hand.idx, hand.now.show_hide);
        }
        if hand.now.toggle_dashboard != hand.before.toggle_dashboard {
            log::debug!(
                "Hand {}: toggle_dashboard {}",
                hand.idx,
                hand.now.toggle_dashboard
            );
        }
        if hand.now.space_drag != hand.before.space_drag {
            log::debug!("Hand {}: space_drag {}", hand.idx, hand.now.space_drag);
        }
        if hand.now.space_rotate != hand.before.space_rotate {
            log::debug!("Hand {}: space_rotate {}", hand.idx, hand.now.space_rotate);
        }
        if hand.now.space_reset != hand.before.space_reset {
            log::debug!("Hand {}: space_reset {}", hand.idx, hand.now.space_reset);
        }
        if hand.now.click_modifier_right != hand.before.click_modifier_right {
            log::debug!(
                "Hand {}: click_modifier_right {}",
                hand.idx,
                hand.now.click_modifier_right
            );
        }
        if hand.now.click_modifier_middle != hand.before.click_modifier_middle {
            log::debug!(
                "Hand {}: click_modifier_middle {}",
                hand.idx,
                hand.now.click_modifier_middle
            );
        }
    }
}

pub struct InteractionState {
    pub mode: PointerMode,
    pub grabbed: Option<GrabData>,
    pub clicked_id: Option<OverlayID>,
    pub hovered_id: Option<OverlayID>,
    pub should_block_input: bool,
    pub should_block_poses: bool,
    pub kbd_block_activated: bool,
}

impl Default for InteractionState {
    fn default() -> Self {
        Self {
            mode: PointerMode::Left,
            grabbed: None,
            clicked_id: None,
            hovered_id: None,
            should_block_input: false,
            should_block_poses: false,
            kbd_block_activated: false,
        }
    }
}

pub struct Pointer {
    pub idx: usize,
    pub pose: Affine3A,
    pub raw_pose: Affine3A,
    pub now: PointerState,
    pub before: PointerState,
    pub last_click: Instant,
    pub pending_haptics: Option<Haptics>,
    pub(super) interaction: InteractionState,
    pub tracked: bool,
    pub handsfree: bool,
    /// This pointer is the mouse ray (see `MouseRay`).
    pub mouse: bool,
}

impl Pointer {
    pub fn new(idx: usize) -> Self {
        debug_assert!(idx == 0 || idx == 1);
        Self {
            idx,
            pose: Affine3A::IDENTITY,
            raw_pose: Affine3A::IDENTITY,
            now: PointerState::default(),
            before: PointerState::default(),
            last_click: Instant::now(),
            pending_haptics: None,
            interaction: InteractionState::default(),
            tracked: false,
            handsfree: false,
            mouse: false,
        }
    }

    pub const fn hand(&self) -> Option<LeftRight> {
        match self.idx {
            0 => Some(LeftRight::Left),
            1 => Some(LeftRight::Right),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PointerState {
    pub scroll_x: f32,
    pub scroll_y: f32,
    pub click: bool,
    pub grab: bool,
    pub grab_float: bool,
    pub alt_click: bool,
    pub show_hide: bool,
    pub toggle_dashboard: bool,
    pub space_drag: bool,
    pub space_rotate: bool,
    pub space_reset: bool,
    pub click_modifier_right: bool,
    pub click_modifier_middle: bool,
    pub move_mouse: bool,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PointerHit {
    pub pointer: usize,
    pub overlay: OverlayID,
    pub mode: PointerMode,
    pub primary: bool,
    pub uv: Vec2,
}

#[derive(Clone, Copy)]
pub struct Haptics {
    pub intensity: f32,
    pub duration: f32,
    pub frequency: f32,
}

#[derive(Debug, Clone, Copy, Default)]
struct RayHit {
    overlay: OverlayID,
    global_pos: Vec3A,
    local_pos: Vec2,
    dist: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct GrabData {
    pub offset: Affine3A,
    pub grabbed_id: OverlayID,
    pub grab_anchor: bool,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, Default)]
pub enum PointerMode {
    #[default]
    Left,
    Right,
    Middle,
    Special,
}

pub struct PointerLine {
    pub mode: PointerMode,
    pub a: Vec3A,
    pub b: Vec3A,
}

fn populate_lines(
    lines: &mut Vec<PointerLine>,
    pointer: &Pointer,
    maybe_hit: Option<(PointerHit, RayHit)>,
    hmd: &Affine3A,
) {
    let Some((hit, raw_hit)) = maybe_hit else {
        if pointer.mouse && pointer.tracked {
            // the mouse pointer in space, between the panels: a cursor at arm's length
            let pos = pointer.pose.transform_point3a(Vec3A::NEG_Z * 1.5);
            push_crosshair(lines, PointerMode::Left, pos, hmd);
        }
        return;
    };

    if pointer.handsfree {
        push_crosshair(lines, hit.mode, raw_hit.global_pos, hmd);
    } else {
        lines.push(PointerLine {
            mode: hit.mode,
            a: pointer.pose.translation,
            b: raw_hit.global_pos,
        });
    }
}

fn push_crosshair(lines: &mut Vec<PointerLine>, mode: PointerMode, pos: Vec3A, hmd: &Affine3A) {
    const HALF_SIZE: f32 = 0.01;

    // horizontal arm
    lines.push(PointerLine {
        mode,
        a: pos - (hmd.x_axis * HALF_SIZE),
        b: pos + (hmd.x_axis * HALF_SIZE),
    });

    // vertical arm
    lines.push(PointerLine {
        mode: PointerMode::Special,
        a: pos - (hmd.y_axis * HALF_SIZE),
        b: pos + (hmd.y_axis * HALF_SIZE),
    });
}

fn update_focus(app: &mut AppState, overlay_input_focus: Option<InputFocus>) {
    if let Some(f) = &overlay_input_focus
        && app.input_state.picking_focus.is_none()
        && app
            .hid_provider
            .set_input_focus(app.wvr_server.as_mut(), *f)
    {
        log::debug!("Setting keyboard focus to {:?}", *f);
    }
}

pub fn interact<O>(
    overlays: &mut OverlayWindowManager<O>,
    app: &mut AppState,
    lines: &mut Vec<PointerLine>,
) -> [Option<Haptics>; 2]
where
    O: Default,
{
    app.input_state.edit_mode = overlays.get_edit_mode();

    // Meta+Alt held: after a moment (so a quick shortcut does not nudge anything) the anchor is
    // fixed to the head, and every anchored panel follows it, keeping its place in the group.
    // Panels placed freely in the room (the page regions shown as panels of their own) make the
    // same move, so the whole arrangement is carried as one rigid group.
    if let Some((since, rel)) = app.input_state.carry.as_mut()
        && since.elapsed() > Duration::from_millis(250)
    {
        let hmd = app.input_state.hmd;
        let rel = *rel.get_or_insert_with(|| hmd.inverse() * app.anchor);
        let before = app.anchor;
        app.anchor = hmd * rel;
        let moved = app.anchor * before.inverse();
        for overlay in overlays.values_mut() {
            let Some(state) = overlay.config.active_state.as_mut() else {
                continue;
            };
            match state.positioning {
                Positioning::Anchored => overlay.config.reset(app, false),
                Positioning::Floating if state.grabbable => {
                    state.transform = moved * state.transform;
                    window::save_transform(state, app);
                    overlay.config.dirty = true;
                }
                _ => {}
            }
        }
    }

    // Meta+Alt+wheel: the screens are laid out again on a circle of a new radius around the head,
    // 8 % a notch (positive = nearer). Every screen keeps its size and faces the head, so on a
    // wider circle the side screens come in towards the middle and on a tighter one they fan
    // out: the group stays side by side around the wearer, wherever the head is now.
    // Meta+Alt+Up / Down: the group grows or shrinks, 8 % a step, and is laid out again at the
    // radius it has, so no gaps open or close.
    let push_all = std::mem::take(&mut app.input_state.mouse_ray.push_all);
    let scale_all = std::mem::take(&mut app.input_state.mouse_ray.scale_all);
    if push_all.abs() > 0.1 || scale_all.abs() > 0.1 {
        let hmd = app.input_state.hmd;
        let head = hmd.translation;
        let before = window::screens_distance(overlays.values(), head);

        if scale_all.abs() > 0.1 {
            let scale = app.anchor.matrix3.x_axis.length();
            let new_scale = (scale * 1.08f32.powf(scale_all)).clamp(0.3, 4.0);
            app.anchor.matrix3 = app.anchor.matrix3.mul_scalar(new_scale / scale);
            if let Some((_, Some(rel))) = app.input_state.carry.as_mut() {
                *rel = hmd.inverse() * app.anchor; // carried at the same time: keep the new size
            }
            for overlay in overlays.values_mut() {
                if overlay
                    .config
                    .active_state
                    .as_ref()
                    .is_some_and(|s| matches!(s.positioning, Positioning::Anchored))
                {
                    overlay.config.reset(app, false);
                }
            }
        }

        if let Some(before) = before {
            let wanted = (before * (1.0 - 0.08 * push_all)).clamp(0.4, 6.0);
            let anchor = app.anchor;
            if let Some(radius) =
                window::layout_screens_arc(overlays.values_mut(), &anchor, head, Some(wanted))
            {
                app.input_state.screen_radius = Some(radius);
                // panels placed freely in the room (the page regions shown as panels of their
                // own) keep their direction and size and go to the new distance with the screens
                let k = radius / before;
                for overlay in overlays.values_mut() {
                    if let Some(state) = overlay.config.active_state.as_mut()
                        && matches!(state.positioning, Positioning::Floating)
                        && state.grabbable
                    {
                        state.transform.translation =
                            head + k * (state.transform.translation - head);
                        window::save_transform(state, app);
                        overlay.config.dirty = true;
                    }
                }
            }
        }
    }

    let hits = if app.input_state.pointers[1].last_click > app.input_state.pointers[0].last_click {
        let right = interact_hand(1, overlays, app);
        let left = interact_hand(0, overlays, app);
        [left, right]
    } else {
        let left = interact_hand(0, overlays, app);
        let right = interact_hand(1, overlays, app);
        [left, right]
    };

    for (idx, hit) in hits.iter().enumerate() {
        populate_lines(
            lines,
            &app.input_state.pointers[idx],
            hit.0,
            &app.input_state.hmd,
        );
    }

    [hits[0].1, hits[1].1]
}

fn interact_hand<O>(
    idx: usize,
    overlays: &mut OverlayWindowManager<O>,
    app: &mut AppState,
) -> (Option<(PointerHit, RayHit)>, Option<Haptics>)
where
    O: Default,
{
    let mut pointer = &mut app.input_state.pointers[idx];
    let pending_haptics = pointer.pending_haptics.take();

    if !pointer.tracked {
        pointer.interaction.should_block_input = false;
        pointer.interaction.should_block_poses = false;
        pointer.interaction.kbd_block_activated = false;
        return (None, pending_haptics); // no hit
    }

    // already grabbing, ignore everything else
    if let Some(grab_data) = pointer.interaction.grabbed {
        if let Some(grabbed) = overlays.mut_by_id(grab_data.grabbed_id) {
            handle_grabbed(idx, grabbed, app);
        } else {
            log::warn!("Grabbed overlay {:?} does not exist", grab_data.grabbed_id);
            pointer.interaction.grabbed = None;
        }
        return (None, pending_haptics);
    }

    let hovered_id = pointer.interaction.hovered_id.take();
    let (Some((mut hit, raw_hit)), haptics) = get_nearest_hit(idx, overlays, app) else {
        handle_no_hit(idx, hovered_id, overlays, app);
        return (None, pending_haptics); // no hit
    };

    // focus change
    if let Some(hovered_id) = hovered_id
        && hovered_id != hit.overlay
        && let Some(old_hovered) = overlays.mut_by_id(hovered_id)
    {
        if old_hovered.primary_pointer.is_some_and(|i| i == idx) {
            old_hovered.primary_pointer = None;
        }
        log::debug!("{} on_left (focus changed)", old_hovered.config.name);
        old_hovered.config.backend.on_left(app, idx);
        old_hovered.hover_pointers[idx] = false;
        if !old_hovered.hover_pointers.iter().any(|x| *x) {
            overlays.edit_overlay(hovered_id, false, app);
        }
    }

    overlays.edit_overlay(hit.overlay, true, app);
    let edit_mode = overlays.get_edit_mode();

    let Some(hovered) = overlays.mut_by_id(hit.overlay) else {
        log::warn!("Hit overlay {:?} does not exist", hit.overlay);
        return (None, pending_haptics); // no hit
    };
    pointer = &mut app.input_state.pointers[idx];
    pointer.interaction.hovered_id = Some(hit.overlay);
    hovered.hover_pointers[idx] = true;

    if let Some(primary_pointer) = hovered.primary_pointer {
        if hit.pointer <= primary_pointer {
            hovered.primary_pointer = Some(hit.pointer);
            hit.primary = true;
        }
    } else {
        hovered.primary_pointer = Some(hit.pointer);
        hit.primary = true;
    }

    if let Some(state) = hovered.config.active_state.as_ref() {
        pointer.interaction.should_block_input = state.block_input
            && (hovered.config.name.as_ref() != WATCH_NAME
                || !app.session.config.block_game_input_ignore_watch);

        pointer.interaction.should_block_poses = state.block_input
            && app.session.config.block_poses_on_kbd_interaction
            && hovered.config.category == OverlayCategory::Keyboard
            && pointer.interaction.kbd_block_activated;
    } else {
        pointer.interaction.should_block_input = false;
        pointer.interaction.should_block_poses = false;
    }

    #[cfg(debug_assertions)]
    log::trace!("Hit: {} {:?}", hovered.config.name, hit);

    let hovered_state = hovered.config.active_state.as_mut().unwrap();

    let grab_float = pointer.now.grab_float;
    let grab_start = pointer.now.grab && !pointer.before.grab;

    // grab
    if grab_start && hovered_state.grabbable {
        update_focus(app, hovered.config.input_focus);
        start_grab(StartGrabParams {
            idx,
            id: hit.overlay,
            name: hovered.config.name.clone(),
            editing: hovered.config.editing,
            state: hovered_state,
            app,
            edit_mode,
            grab_float,
        });
        log::debug!("Hand {}: grabbed {}", hit.pointer, hovered.config.name);
        return (
            Some((hit, raw_hit)),
            Some(Haptics {
                intensity: 0.25,
                duration: 0.1,
                frequency: 0.1,
            }),
        );
    }

    handle_scroll(&hit, hovered, app);

    // click / release
    let pointer = &mut app.input_state.pointers[hit.pointer];
    if pointer.now.click && !pointer.before.click {
        pointer.interaction.clicked_id = Some(hit.overlay);
        if hovered.config.category == OverlayCategory::Keyboard {
            pointer.interaction.kbd_block_activated = true;
        }
        update_focus(app, hovered.config.input_focus);
        hovered.config.backend.on_pointer(app, &hit, true);
    } else if !pointer.now.click && pointer.before.click {
        // send release event to overlay that was originally clicked
        if let Some(clicked_id) = pointer.interaction.clicked_id.take() {
            if let Some(clicked) = overlays.mut_by_id(clicked_id) {
                clicked.config.backend.on_pointer(app, &hit, false);
            }
        } else {
            hovered.config.backend.on_pointer(app, &hit, false);
        }
    }

    (Some((hit, raw_hit)), haptics.or(pending_haptics))
}

fn handle_no_hit<O>(
    pointer_idx: usize,
    hovered_id: Option<OverlayID>,
    overlays: &mut OverlayWindowManager<O>,
    app: &mut AppState,
) {
    if let Some(hovered_id) = hovered_id
        && let Some(hovered) = overlays.mut_by_id(hovered_id)
    {
        log::debug!("{} on_left (no hit)", hovered.config.name);
        hovered.config.backend.on_left(app, pointer_idx);
        hovered.hover_pointers[pointer_idx] = false;
        if !hovered.hover_pointers.iter().any(|x| *x) {
            overlays.edit_overlay(hovered_id, false, app);
        }
    }

    let pointer = &mut app.input_state.pointers[pointer_idx];
    pointer.interaction.should_block_input = false;
    pointer.interaction.should_block_poses = false;
    pointer.interaction.kbd_block_activated = false;

    // in case click released while not aiming at anything
    // send release event to overlay that was originally clicked
    if !pointer.now.click
        && pointer.before.click
        && let Some(clicked_id) = pointer.interaction.clicked_id.take()
        && let Some(clicked) = overlays.mut_by_id(clicked_id)
    {
        let hit = PointerHit {
            pointer: pointer.idx,
            overlay: clicked_id,
            mode: pointer.interaction.mode,
            ..Default::default()
        };
        clicked.config.backend.on_pointer(app, &hit, false);
    }
}

fn handle_scroll<O>(hit: &PointerHit, hovered: &mut OverlayWindowData<O>, app: &mut AppState) {
    let pointer = &mut app.input_state.pointers[hit.pointer];
    if pointer.now.scroll_x.abs() <= 0.1 && pointer.now.scroll_y.abs() <= 0.1 {
        return;
    }

    if pointer.mouse && app.input_state.edit_mode {
        // edit mode with the mouse: the wheel brings the panel nearer (wheel up) or pushes it away
        let (hmd, notches) = (app.input_state.hmd, pointer.now.scroll_y);
        if let Some(state) = hovered.config.active_state.as_mut() {
            push_pull_from_head(&mut state.transform, &hmd, notches);
            window::save_transform(state, app);
        }
        // without this the moved panel is not handed to the compositor
        hovered.config.dirty = true;
        return;
    }

    let config = &app.session.config;
    let scroll_factor = 90.0 * app.delta_time;

    let scroll_x = pointer.now.scroll_x
        * scroll_factor
        * config.scroll_speed
        * if config.invert_scroll_direction_x {
            -1.0
        } else {
            1.0
        };
    let scroll_y = pointer.now.scroll_y
        * scroll_factor
        * config.scroll_speed
        * if config.invert_scroll_direction_x {
            -1.0
        } else {
            1.0
        };

    if app.input_state.pointers[1 - hit.pointer]
        .interaction
        .grabbed
        .is_some_and(|x| x.grabbed_id == hit.overlay)
    {
        let can_curve = hovered
            .frame_meta()
            .is_some_and(|e| e.extent[0] >= e.extent[1]);

        // re-borrow
        let hovered_state = hovered.config.active_state.as_mut().unwrap();
        if can_curve {
            let cur = hovered_state.curvature.unwrap_or(0.0);
            let new = scroll_y.mul_add(-0.01, cur).min(0.5);
            if new <= f32::EPSILON {
                hovered_state.curvature = None;
            } else {
                hovered_state.curvature = Some(new);
            }
        } else {
            hovered_state.curvature = None;
        }
    } else {
        hovered.config.backend.on_scroll(
            app,
            hit,
            WheelDelta {
                x: scroll_x,
                y: scroll_y,
            },
        );
    }
}

fn get_nearest_hit<O>(
    pointer_idx: usize,
    overlays: &mut OverlayWindowManager<O>,
    app: &mut AppState,
) -> (Option<(PointerHit, RayHit)>, Option<Haptics>)
where
    O: Default,
{
    let pointer = &mut app.input_state.pointers[pointer_idx];
    if !pointer.tracked {
        return (None, None);
    }

    let ray_origin = pointer.pose;
    let mode = pointer.interaction.mode;
    let edit_mode = overlays.get_edit_mode();

    let mut hits: SmallVec<[RayHit; 8]> = smallvec!();

    for (id, overlay) in overlays.iter() {
        let Some(overlay_state) = overlay.config.active_state.as_ref() else {
            continue;
        };
        if !overlay_state.interactable && !edit_mode {
            continue;
        }

        if let Some(hit) = ray_test(
            &ray_origin,
            id,
            &overlay_state.transform,
            overlay_state.curvature.as_ref(),
        ) && hit.dist.is_finite()
        {
            hits.push(hit);
        }
    }

    hits.sort_by(|a, b| a.dist.total_cmp(&b.dist));

    for hit in hits {
        let overlay = overlays.mut_by_id(hit.overlay).unwrap(); // safe because we just got the id from the overlay

        let Some(uv) = overlay
            .config
            .backend
            .as_mut()
            .get_interaction_transform()
            .map(|a| a.transform_point2(hit.local_pos))
        else {
            continue;
        };

        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) && !overlay.config.resizing {
            continue;
        }

        let pointer_hit = PointerHit {
            pointer: pointer_idx,
            overlay: hit.overlay,
            mode,
            primary: false,
            uv,
        };

        let result = overlay.config.backend.on_hover(app, &pointer_hit);
        if result.consume || overlay.config.editing {
            return (Some((pointer_hit, hit)), result.haptics);
        }
    }

    (None, None)
}

struct StartGrabParams<'a> {
    idx: usize,
    id: OverlayID,
    name: Arc<str>,
    editing: bool,
    state: &'a mut OverlayWindowState,
    app: &'a mut AppState,
    edit_mode: bool,
    grab_float: bool,
}

fn start_grab(par: StartGrabParams) {
    let (app, id, state) = (par.app, par.id, par.state);

    let pointer = &mut app.input_state.pointers[par.idx];

    // Grab anchor if:
    // - grabbed overlay is Anchored
    // - not in editmode
    // - not using grab_float
    // - grabbing with one hand. (grabbing with the 2nd hand will grab the individual overlay instead)
    let grab_anchor = !par.edit_mode
        && !par.grab_float
        && !app.anchor_grabbed
        && matches!(state.positioning, Positioning::Anchored);

    let relative_grab_transform = if grab_anchor {
        app.anchor
    } else {
        state.transform
    };

    let offset = pointer.pose.inverse() * relative_grab_transform;

    app.anchor_grabbed = grab_anchor;

    pointer.interaction.grabbed = Some(GrabData {
        offset,
        grabbed_id: id,
        grab_anchor,
    });

    app.tasks.enqueue(TaskType::Overlay(OverlayTask::Modify(
        OverlaySelector::Id(id),
        Box::new({
            let pos = state.positioning;
            let name = par.name.clone();
            move |app, o| {
                let _ = o
                    .backend
                    .notify(
                        app,
                        OverlayEventData::OverlayGrabbed {
                            name,
                            pos,
                            editing: par.editing,
                        },
                    )
                    .inspect_err(|e| log::warn!("Error during Notify OverlayGrabbed: {e:?}"));
            }
        }),
    )));

    // Show anchor
    app.tasks.enqueue(TaskType::Overlay(OverlayTask::Modify(
        OverlaySelector::Name(ANCHOR_NAME.clone()),
        Box::new(|app, o| {
            o.activate(app, true);
        }),
    )));

    if let Some(hand) = pointer.hand()
        && !app.session.config.hide_grab_help
        && !pointer.handsfree
    {
        let pos = state.positioning;
        app.tasks.enqueue(TaskType::Overlay(OverlayTask::Modify(
            OverlaySelector::Name(GRAB_HELP_NAME.clone()),
            Box::new(move |app, o| {
                let _ = o
                    .backend
                    .notify(
                        app,
                        OverlayEventData::OverlayGrabbed {
                            name: par.name,
                            pos,
                            editing: par.editing,
                        },
                    )
                    .inspect_err(|e| log::warn!("Error during Notify OverlayGrabbed: {e:?}"));

                o.default_state.positioning = Positioning::FollowHand { hand, lerp: 0.1 };
                o.default_state.align_to_hmd = true;
                o.activate(app, true);
            }),
        )));
    }
}

/// Move `transform` along the line from the head by `notches` wheel notches, each 8 % of the
/// distance (positive = nearer), staying between 0.3 and 6 m.
fn push_pull_from_head(transform: &mut Affine3A, hmd: &Affine3A, notches: f32) {
    let from_head = transform.translation - hmd.translation;
    let dist = from_head.length();
    if dist < 1e-3 || !notches.is_finite() {
        return;
    }
    let new_dist = (dist * (1.0 - 0.08 * notches)).clamp(0.3, 6.0);
    transform.translation = hmd.translation + from_head * (new_dist / dist);
}

fn handle_scale(transform: &mut Affine3A, scroll_y: f32) {
    let cur_scale = transform.x_axis.length();
    if cur_scale < 0.1 && scroll_y > 0.0 {
        return;
    }
    if cur_scale > 20. && scroll_y < 0.0 {
        return;
    }

    transform.matrix3 = transform
        .matrix3
        .mul_scalar(0.025f32.mul_add(-scroll_y, 1.0));
}

#[allow(clippy::too_many_lines)]
/// A grabbed panel follows the mouse ray; the wheel moves it along the ray (notches, positive =
/// nearer, 8 % of the distance each), between 0.3 and 6 m.
fn mouse_push_pull(offset: &mut Affine3A, notches: f32) {
    if notches.abs() < 0.1 || !notches.is_finite() {
        return;
    }
    let dist = (-offset.translation.z).max(0.3);
    offset.translation.z = -(dist * (1.0 - 0.08 * notches)).clamp(0.3, 6.0);
}

fn handle_grabbed<O>(idx: usize, overlay: &mut OverlayWindowData<O>, app: &mut AppState)
where
    O: Default,
{
    let pointer = &mut app.input_state.pointers[idx];
    let Some(grab_data) = pointer.interaction.grabbed.as_mut() else {
        log::error!("Grabbed overlay does not exist");
        return;
    };
    let grab_anchor = grab_data.grab_anchor;

    if pointer.now.grab {
        let Some(overlay_state) = overlay.config.active_state.as_mut() else {
            // overlay got toggled off while being grabbed. those dastardly users!
            // just wait for them to release the grab
            return;
        };

        let scroll_factor = 90.0 * app.delta_time;

        if grab_anchor {
            if pointer.now.click {
                pointer.interaction.mode = PointerMode::Special;
                let grab_dist = grab_data.offset.translation.length().clamp(0.5, 5.0) * 0.2 + 0.4;
                handle_scale(
                    &mut app.anchor,
                    pointer.now.scroll_y * grab_dist * scroll_factor,
                );
                handle_scale(
                    &mut grab_data.offset,
                    pointer.now.scroll_y * grab_dist * scroll_factor,
                );
            } else if pointer.mouse {
                mouse_push_pull(&mut grab_data.offset, pointer.now.scroll_y);
            } else if app.session.config.allow_sliding && pointer.now.scroll_y.is_finite() {
                // single grab push/pull
                let grab_dist = grab_data.offset.translation.length().clamp(0.5, 5.0);
                grab_data.offset.translation.z -=
                    pointer.now.scroll_y * scroll_factor * 0.02 * grab_dist;
                grab_data.offset.translation.z = grab_data.offset.translation.z.min(-0.05);
            }
            if pointer.now.click_modifier_right {
                app.anchor = pointer.pose * grab_data.offset;
            } else {
                app.anchor.translation =
                    pointer.pose.transform_point3a(grab_data.offset.translation);
                let scale = scalar_scale(&app.anchor);
                realign(
                    &mut app.anchor,
                    &app.input_state.hmd,
                    scale,
                    app.session.config.snap_angle_deg,
                );
            }
        } else {
            // single grab resize
            if pointer.now.click {
                pointer.interaction.mode = PointerMode::Special;
                let grab_dist = grab_data.offset.translation.length().clamp(0.5, 5.0) * 0.2 + 0.4;
                handle_scale(
                    &mut overlay_state.transform,
                    pointer.now.scroll_y * grab_dist * scroll_factor,
                );
                handle_scale(
                    &mut grab_data.offset,
                    pointer.now.scroll_y * grab_dist * scroll_factor,
                );
            } else if pointer.mouse {
                mouse_push_pull(&mut grab_data.offset, pointer.now.scroll_y);
            } else if app.session.config.allow_sliding && pointer.now.scroll_y.is_finite() {
                // single grab push/pull
                let grab_dist = grab_data.offset.translation.length().clamp(0.5, 5.0);
                grab_data.offset.translation.z -=
                    pointer.now.scroll_y * scroll_factor * 0.02 * grab_dist;
                grab_data.offset.translation.z = grab_data.offset.translation.z.min(-0.05);
            }
            if pointer.now.click_modifier_right {
                overlay_state.transform = pointer.pose * grab_data.offset;
            } else {
                let scale = scalar_scale(&overlay_state.transform);
                overlay_state.transform.translation =
                    pointer.pose.transform_point3a(grab_data.offset.translation);
                realign_roll(
                    &mut overlay_state.transform,
                    &app.input_state.hmd,
                    scale,
                    app.session.config.snap_angle_deg,
                    pointer.mouse, // dragged with the mouse: the side tilt is the head's
                );
            }
            overlay.config.pause_movement = true;
            overlay.config.dirty = true;
        }
    } else {
        // not now.grab
        pointer.interaction.grabbed = None;
        if grab_anchor {
            app.anchor_grabbed = false;
        } else {
            // single grab released
            if &*overlay.config.name == WATCH_NAME {
                // watch special: when dropped, follow the hand that wasn't grabbing
                if let Some(overlay_state) = overlay.config.active_state.as_mut() {
                    let align_to_hmd = overlay_state.align_to_hmd;
                    overlay_state.positioning = match overlay_state.positioning {
                        Positioning::FollowHand { hand, lerp } => match pointer.hand() {
                            Some(LeftRight::Left) => Positioning::FollowHand {
                                hand: LeftRight::Right,
                                lerp,
                            },
                            Some(LeftRight::Right) => Positioning::FollowHand {
                                hand: LeftRight::Left,
                                lerp,
                            },
                            _ => Positioning::FollowHand { hand, lerp },
                        },
                        x => x,
                    };
                    overlay_state.align_to_hmd = align_to_hmd;
                }
            } else if overlay.config.global
                && let Some(active_state) = overlay.config.active_state.as_ref()
            {
                let cur_scale = overlay.config.default_state.transform.x_axis.length();
                let tgt_scale = active_state.transform.x_axis.length();
                let mat = &mut overlay.config.default_state.transform.matrix3;
                *mat = mat.mul_scalar(tgt_scale / cur_scale);
            }
            overlay.config.pause_movement = false;
            if let Some(overlay_state) = overlay.config.active_state.as_mut() {
                window::save_transform(overlay_state, app);
            }
        }

        // Hide anchor
        app.tasks.enqueue(TaskType::Overlay(OverlayTask::Modify(
            OverlaySelector::Name(ANCHOR_NAME.clone()),
            Box::new(|_app, o| {
                o.deactivate();
            }),
        )));
        app.tasks.enqueue(TaskType::Overlay(OverlayTask::Modify(
            OverlaySelector::Name(GRAB_HELP_NAME.clone()),
            Box::new(|_app, o| {
                o.deactivate();
            }),
        )));
        log::debug!("Hand {}: dropped {}", idx, overlay.config.name);
    }
}

fn ray_test(
    ray_origin: &Affine3A,
    overlay: OverlayID,
    overlay_pose: &Affine3A,
    curvature: Option<&f32>,
) -> Option<RayHit> {
    let (dist, local_pos) = curvature.map_or_else(
        || {
            Some(raycast_plane(
                ray_origin,
                Vec3A::NEG_Z,
                overlay_pose,
                Vec3A::NEG_Z,
            ))
        },
        |curvature| raycast_cylinder(ray_origin, Vec3A::NEG_Z, overlay_pose, *curvature),
    )?;

    if dist < 0.0 {
        // hit is behind us
        return None;
    }

    Some(RayHit {
        overlay,
        global_pos: ray_origin.transform_point3a(Vec3A::NEG_Z * dist),
        local_pos,
        dist,
    })
}

fn raycast_plane(
    source: &Affine3A,
    source_fwd: Vec3A,
    plane: &Affine3A,
    plane_norm: Vec3A,
) -> (f32, Vec2) {
    let plane_normal = plane.transform_vector3a(plane_norm);
    let ray_dir = source.transform_vector3a(source_fwd);

    let d = plane.translation.dot(-plane_normal);
    let mut dist = -(d + source.translation.dot(plane_normal)) / ray_dir.dot(plane_normal);

    let hit_local = plane
        .inverse()
        .transform_point3a(source.translation + ray_dir * dist)
        .xy();

    // hitting the backside of the plane, make the hit invalid
    if ray_dir.dot(plane_normal) < 0.0 && dist.is_sign_positive() {
        dist = -dist;
    }

    (dist, hit_local)
}

fn raycast_cylinder(
    source: &Affine3A,
    source_fwd: Vec3A,
    plane: &Affine3A,
    curvature: f32,
) -> Option<(f32, Vec2)> {
    // this is solved locally; (0,0) is the center of the cylinder, and the cylinder is aligned with the Y axis
    let size = plane.x_axis.length();
    let to_local = Affine3A {
        matrix3: plane.matrix3.mul_scalar(1.0 / size),
        translation: plane.translation,
    }
    .inverse();

    let radius = size / (2.0 * PI * curvature);

    let ray_dir = to_local.transform_vector3a(source.transform_vector3a(source_fwd));
    let ray_origin = to_local.transform_point3a(source.translation) + Vec3A::NEG_Z * radius;

    let v_dir = ray_dir.xz();
    let v_pos = ray_origin.xz();

    let l_dir = v_dir.dot(v_dir);
    let l_pos = v_dir.dot(v_pos);
    let c = radius.mul_add(-radius, v_pos.dot(v_pos));

    let d = l_pos.mul_add(l_pos, -(l_dir * c));
    if d < f32::EPSILON {
        return None;
    }

    let sqrt_d = d.sqrt();

    let t1 = (-l_pos - sqrt_d) / l_dir;
    let t2 = (-l_pos + sqrt_d) / l_dir;

    let mut t = t1.max(t2);

    if t < f32::EPSILON {
        return None;
    }

    let mut hit_local = ray_origin + ray_dir * t;
    if hit_local.z > 0.0 {
        // hitting the opposite half of the cylinder
        return None;
    }

    let normal = Vec3A::new(hit_local.x, 0.0, hit_local.z).normalize();
    // If hitting from the outside, flip t
    if ray_dir.dot(normal) < 0.0 && t.is_sign_positive() {
        t = -t;
    }

    let max_angle = 2.0 * (size / (2.0 * radius));
    let x_angle = (hit_local.x / radius).asin();

    hit_local.x = x_angle / max_angle;
    hit_local.y /= size;

    Some((t, hit_local.xy()))
}

/*
 * Pinch grab: a pinch over a panel takes the panel along (Dennis, 2026-11-13).
 */

/// A panel held by a pinch: where its centre sits in the frame of the line of sight through the
/// pinch point, so it follows the hand along that line at its distance.
pub struct PinchGrab {
    pub id: OverlayID,
    pub panel_in_sight: Vec3A,
    /// Distance of the pinch point from the head when the grab started: the panel's distance
    /// scales with the hand's (hand 20 % farther, panel 20 % farther).
    pub hand_dist: f32,
    /// The smoothed scale factor of the moment.
    pub depth_factor: f32,
    /// Frames in a row with the pinch read as open: the drop waits for a few, finger tips jitter.
    pub open_frames: u32,
    pub last_seen: Instant,
}

const PINCH_CLOSE_M: f32 = 0.035; // the tracked finger tips stay ~2 cm apart in a real pinch
const PINCH_OPEN_M: f32 = 0.055;
const PINCH_CONFIRM_FRAMES: u32 = 3;
const PINCH_RELEASE_FRAMES: u32 = 4;
const PINCH_LOST_GRACE: Duration = Duration::from_millis(400);
const PINCH_DEPTH_SMOOTHING: f32 = 0.25; // per frame, against the tracker's depth jitter
const PANEL_DIST_MIN: f32 = 0.3;
const PANEL_DIST_MAX: f32 = 6.0;

/// A frame at the head whose -Z is the line of sight to `target`, upright by the head's up.
fn sight_frame(hmd: &Affine3A, target: Vec3A) -> Option<Affine3A> {
    let fwd = target - hmd.translation;
    let len = fwd.length();
    if len < 0.05 || !len.is_finite() {
        return None;
    }
    let fwd = fwd / len;
    let mut up = hmd.y_axis.normalize();
    if up.cross(fwd).length() < 1e-3 {
        up = Vec3A::Y;
    }
    let right = fwd.cross(up).normalize();
    let up = right.cross(fwd).normalize();
    Some(Affine3A::from_cols(right, up, -fwd, hmd.translation))
}

/// The nearest panel on the line of sight (the hit inside the panel's picture).
fn overlay_under_sight<O>(
    overlays: &mut OverlayWindowManager<O>,
    sight: &Affine3A,
) -> Option<OverlayID>
where
    O: Default,
{
    let edit_mode = overlays.get_edit_mode();
    let mut hits: SmallVec<[RayHit; 8]> = smallvec!();
    for (id, overlay) in overlays.iter() {
        let Some(state) = overlay.config.active_state.as_ref() else {
            continue;
        };
        if (!state.interactable && !edit_mode) || overlay.config.ghost {
            continue;
        }
        if let Some(hit) = ray_test(sight, id, &state.transform, state.curvature.as_ref())
            && hit.dist.is_finite()
        {
            hits.push(hit);
        }
    }
    hits.sort_by(|a, b| a.dist.total_cmp(&b.dist));
    for hit in hits {
        let Some(overlay) = overlays.mut_by_id(hit.overlay) else {
            continue;
        };
        let Some(uv) = overlay
            .config
            .backend
            .as_mut()
            .get_interaction_transform()
            .map(|a| a.transform_point2(hit.local_pos))
        else {
            continue;
        };
        if (0.0..=1.0).contains(&uv.x) && (0.0..=1.0).contains(&uv.y) {
            return Some(hit.overlay);
        }
    }
    None
}

/// A pinch (thumb tip to index tip) over a panel, as seen from the head, grabs the panel. While
/// it is held the panel follows the hand along the line of sight at its distance and faces the
/// head, rolling with the head (as a mouse drag does); letting go leaves it there and saves the
/// place. `pinch` is the pair of tips in stage space while the hand is tracked.
pub fn hand_pinch_grab<O>(
    overlays: &mut OverlayWindowManager<O>,
    app: &mut AppState,
    hand: usize,
    pinch: Option<(Vec3A, Vec3A)>,
) where
    O: Default,
{
    let now = Instant::now();
    let hmd = app.input_state.hmd;
    let (point, gap) = match pinch {
        Some((thumb, index)) => (Some((thumb + index) * 0.5), (thumb - index).length()),
        None => (None, f32::INFINITY),
    };

    if let Some(mut grab) = app.input_state.pinch_grabs[hand].take() {
        let Some(overlay) = overlays.mut_by_id(grab.id) else {
            return; // the panel is gone
        };
        if gap < PINCH_OPEN_M {
            grab.open_frames = 0;
        } else if point.is_some() {
            grab.open_frames += 1;
        }
        let (keep, seen) = match point {
            None => (now - grab.last_seen < PINCH_LOST_GRACE, false), // lost for a moment: hold on
            Some(_) if grab.open_frames >= PINCH_RELEASE_FRAMES => (false, false),
            Some(p) => {
                if let (Some(sight), Some(state)) =
                    (sight_frame(&hmd, p), overlay.config.active_state.as_mut())
                {
                    // depth: the hand's distance from the head, relative to the grab, moves the
                    // panel along the line of sight by the same ratio
                    let hand_dist = (p - hmd.translation).length();
                    let wanted = (hand_dist / grab.hand_dist).clamp(0.25, 4.0);
                    grab.depth_factor += (wanted - grab.depth_factor) * PINCH_DEPTH_SMOOTHING;
                    let panel_dist = grab.panel_in_sight.length();
                    let factor = if panel_dist > 1e-3 {
                        (grab.depth_factor * panel_dist).clamp(PANEL_DIST_MIN, PANEL_DIST_MAX)
                            / panel_dist
                    } else {
                        1.0
                    };
                    state.transform.translation =
                        sight.transform_point3a(grab.panel_in_sight * factor);
                    let scale = scalar_scale(&state.transform);
                    realign_roll(
                        &mut state.transform,
                        &hmd,
                        scale,
                        app.session.config.snap_angle_deg,
                        true,
                    );
                    overlay.config.dirty = true;
                }
                (true, true)
            }
        };
        if keep {
            if seen {
                grab.last_seen = now;
            }
            app.input_state.pinch_grabs[hand] = Some(grab);
        } else {
            overlay.config.pause_movement = false;
            if let Some(state) = overlay.config.active_state.as_mut() {
                window::save_transform(state, app);
            }
            overlay.config.dirty = true;
            log::info!("Pinch {hand}: dropped {}", overlay.config.name);
        }
        return;
    }

    // Nothing held: a pinch closed for a few frames over a panel starts a grab.
    let Some(p) = point else {
        app.input_state.pinch_closed[hand] = 0;
        return;
    };
    if gap < PINCH_CLOSE_M {
        app.input_state.pinch_closed[hand] += 1;
    } else {
        app.input_state.pinch_closed[hand] = 0;
        return;
    }
    if app.input_state.pinch_closed[hand] < PINCH_CONFIRM_FRAMES {
        return;
    }
    let Some(sight) = sight_frame(&hmd, p) else {
        return;
    };
    let Some(id) = overlay_under_sight(overlays, &sight) else {
        return;
    };
    let Some(overlay) = overlays.mut_by_id(id) else {
        return;
    };
    let Some(state) = overlay.config.active_state.as_ref() else {
        return;
    };
    let panel_in_sight = sight
        .inverse()
        .transform_point3a(state.transform.translation);
    let hand_dist = (p - hmd.translation).length().max(0.05);
    overlay.config.pause_movement = true;
    app.input_state.pinch_closed[hand] = 0;
    app.input_state.pinch_grabs[hand] = Some(PinchGrab {
        id,
        panel_in_sight,
        hand_dist,
        depth_factor: 1.0,
        open_frames: 0,
        last_seen: now,
    });
    log::info!("Pinch {hand}: grabbed {}", overlay.config.name);
}
