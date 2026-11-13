# Branch `xr-desk`: what this fork adds

WayVR, as changed for a desk in see-through glasses: the wearer sits at a real keyboard and
mouse in front of real monitors, and WayVR shows the same screens as panels in space. This
page lists what the branch adds to upstream and where each piece lives. Everything is off or
inert unless noted, so the branch still runs as plain WayVR.

Build: `cargo build --release -p wayvr --no-default-features --features openxr,wayland,x11`

## Screens

| Addition | Where |
|---|---|
| **Screens straight from KWin.** On KDE every output's picture stream is requested from KWin (`zkde_screencast_unstable_v1`), not through the portal: no share dialog, and a panel cannot show another monitor than its own. The portal remains the fallback | `wlx-capture/src/kde_screencast.rs`, `wayvr/src/overlays/screen/wl.rs` |
| **Unknown monitors are shown.** A screen that appears nowhere in the saved arrangement (new, or its connector got another number) is shown at its default place | `wayvr/src/windowing/manager.rs` (`restore_layout`) |
| **The desktop cursor's own picture**, taken from the stream's cursor metadata and drawn by the screen pipeline, so it moves while the picture stands still | `wlx-capture/src/pipewire/capture.rs`, `wayvr/src/overlays/screen/` |
| **Black as see-through** (`WAYVR_BLACK_KEY=1`): black pixels of a mirrored screen let what is behind show through | `wayvr/src/shaders/`, screen pipeline |
| **The on-screen keyboard starts hidden**: a real keyboard is in use | `wayvr/src/windowing/manager.rs` |

## Arrangement

| Addition | Where |
|---|---|
| **All screens in one set on a circle around the head**, side by side in desktop order | `wayvr/src/overlays/screen/wl.rs`, `wayvr/src/windowing/window.rs` (`layout_screens_arc`) |
| **Meta+Alt+wheel**: a new radius, 8 % a notch, 0.4 to 6 m. Screens keep their size, are placed again on the circle around where the head is, face it and are bent to it. Freely placed panels go to the new distance along their own line | `wayvr/src/backend/input.rs` |
| **Meta+Alt+Up / Down**: all screens larger or smaller, 8 % a step, then laid out again | `wayvr/src/backend/input.rs` |
| **Hold Meta+Alt**: the arrangement, freely placed panels included, is carried with the head | `wayvr/src/backend/input.rs` |
| **`wayvrctl reset-screens`**: the screens back on the circle in front of the head, at the radius last chosen | `wayvr/src/ipc/`, `wayvr/src/windowing/manager.rs` |

These keys are read from the keyboard devices all the time, without taking them from the
desktop (`wayvr/src/backend/wayvr/input_capture.rs`).

## Mouse and keyboard

| Addition | Where |
|---|---|
| **Captured only for edit mode** (Meta+Alt+E): a pointer in space steered by the mouse from the head, the left button drags a panel, the wheel moves it nearer or farther, head roll tilts it | `wayvr/src/backend/wayvr/`, `wayvr/src/backend/input.rs` |
| **A hook on grab and release** (`WAYVR_CAPTURE_HOOK=<program>`): the program is run with `grabbed` or `ungrabbed`, also for a grab made over IPC | `wayvr/src/backend/wayvr/mod.rs` |

## Hands

| Addition | Where |
|---|---|
| **Hand skeletons**: the tracked hands (`XR_EXT_hand_tracking`) drawn as bright lines | `wayvr/src/backend/openxr/mod.rs`, `wayvr/src/backend/openxr/lines.rs` |
| **Pinch grab**: a pinch in front of a panel takes it along; the hand's distance from the head moves it along the line of sight; the head's roll tilts it. Off with `WAYVR_PINCH_GRAB=0`. Thresholds: 3.0 cm to grab, 4.0 cm to release, from a labelled measurement (constants at `pinch_thresholds`); `WAYVR_PINCH_CM="<grab> <release>"` overrides them | `wayvr/src/backend/input.rs` |

## For other programs

| Addition | Where |
|---|---|
| **Layout server** (`WAYVR_LAYOUT_PORT=<port>`): head pose and every panel's pose, size, curvature and desktop rectangle as JSON on `http://127.0.0.1:<port>/layout`; region requests are posted to it. `WAYVR_LAYOUT_FILE=1` writes the same to `$XDG_RUNTIME_DIR/wayvr-layout.json` | `wayvr/src/backend/openxr/layout_file.rs` |
| **Region panels**: a rectangle of a mirrored screen as a panel of its own (`region:<id>`), with a shape, pop-ups attached to their parent, and the source screen optionally hidden while it keeps updating. Used by a browser extension to lift parts of a page into space | `wayvr/src/overlays/region.rs` |
| **Depth occlusion** (`WAYVR_DEPTH_OCCLUSION=1`): screens are cut out where a depth map (`/dev/shm/viture-depth`) says a real surface is nearer; the mask is warped to the eyes' viewpoint with the camera pose from the calibration store (`WAYVR_DEPTH_CAMERA` overrides the file) | `wayvr/src/overlays/occlusion.rs`, `wayvr/src/shaders/` |
| **Calibration board overlay**: a printed ChArUco board drawn where a camera sees it, from a pose file (`$XDG_RUNTIME_DIR/xr-calib-board.json`), as a look-and-see test of a calibration | `wayvr/src/overlays/calibboard.rs`, `shaders/checker.frag` |
| **Relief panel** (experiment, `WAYVR_RELIEF`): the 3D view of a page as a surface in space | `wayvr/src/overlays/relief.rs` |

## Notes for whoever changes this

- Panels are either anchored (placed relative to one anchor that a recentre moves) or floating
  (placed in the room). Screens are anchored; region panels float. Code that moves "everything"
  has to handle both.
- The arrangement is saved under the panels' names. Screen panels are named after outputs,
  and output names change; never rely on a saved name existing.
- The portal lets the user pick any screen, whatever was asked for. Where the portal is still
  used, the mouse is mapped to the screen that was actually returned, and a wrong pick is not
  remembered.
- Scaling everything about the head changes nothing visible but depth. A distance change has
  to keep the panels' size.
