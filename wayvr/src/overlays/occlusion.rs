//! Real things in front of a screen hide it: a mask from the glasses' depth map.
//!
//! A tracker outside this program publishes a metric depth map of the left tracking camera in
//! shared memory (`/dev/shm/viture-depth`, block `VDP1`). For a screen, [`mask`] puts every point
//! of the map into the room as the camera stood when the map was taken, and carries it along
//! the line from the eye onto the screen: where the room is nearer than the screen on that
//! line, the mask is opaque black, which is see-through on additive glasses. The screen
//! backend draws the mask over its picture with the cursor pipeline.
//!
//! Off unless `WAYVR_DEPTH_OCCLUSION=1`. The camera's place in the head is
//! `cameras.left.pose_in_head` of the XR desk's calibration store
//! (`~/.config/xr-desk/calibration.json`), or, without a store, `depth-camera.json` beside it,
//! or the file named by `WAYVR_DEPTH_CAMERA`; the last two hold the pose alone:
//! `{"r": [9 numbers, row-major], "t": [3]}` with `p_head = r * p_camera + t`, the camera in
//! OpenCV axes (x right, y down, z forward), the head in OpenXR view axes. Without any of them
//! the camera sits in the head's origin looking forward. The file is read again when it changes.

use std::{
    collections::VecDeque,
    fs::File,
    os::unix::fs::FileExt,
    sync::{LazyLock, Mutex},
    time::SystemTime,
};

use glam::{Affine3A, Mat3A, Vec3A};

/// Grid of the mask over a screen.
pub const MASK_W: u32 = 192;
pub const MASK_H: u32 = 108;
/// Width of the soft edge as a fraction of the distance.
const SIGMA: f32 = 0.08;
/// How often the camera file is looked at for a change, seconds.
const CAMERA_CHECK_S: f64 = 0.5;
/// How long a depth map may be used after it was taken, seconds.
const MAX_AGE: f64 = 0.5;
const HEADER: usize = 128;

pub fn enabled() -> bool {
    static ON: LazyLock<bool> =
        LazyLock::new(|| std::env::var("WAYVR_DEPTH_OCCLUSION").is_ok_and(|v| v != "0"));
    *ON
}

struct Depth {
    count: u64,
    w: usize,
    h: usize,
    t_taken: f64,
    f: f32,
    /// rectified frame <- left camera
    rot: Mat3A,
    inv: Vec<f32>,
}

struct State {
    file: Option<File>,
    depth: Option<Depth>,
    /// head <- left camera (OpenCV axes)
    cam_r: Mat3A,
    cam_t: Vec3A,
    /// The camera file's modification time when it was last read, and when that was looked at.
    cam_mtime: Option<SystemTime>,
    cam_checked: f64,
    /// (CLOCK_MONOTONIC seconds, head pose in the stage space)
    poses: VecDeque<(f64, Affine3A)>,
}

static STATE: LazyLock<Mutex<State>> = LazyLock::new(|| {
    // OpenCV camera axes to OpenXR view axes, the camera in the head's origin
    let (cam_r, cam_t) = load_camera().unwrap_or((
        Mat3A::from_cols(Vec3A::X, Vec3A::NEG_Y, Vec3A::NEG_Z),
        Vec3A::ZERO,
    ));
    Mutex::new(State {
        file: None,
        depth: None,
        cam_r,
        cam_t,
        cam_mtime: camera_mtime(),
        cam_checked: monotonic(),
        poses: VecDeque::new(),
    })
});

fn monotonic() -> f64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: plain libc call with a valid pointer
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &raw mut ts) };
    ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9
}

/// The file with the camera's place in the head, and whether it is the calibration store.
///
/// `WAYVR_DEPTH_CAMERA` names a file of the plain form; otherwise the XR desk's calibration
/// store when it exists (`cameras.left.pose_in_head`), otherwise `depth-camera.json`.
fn camera_path() -> (String, bool) {
    if let Ok(path) = std::env::var("WAYVR_DEPTH_CAMERA") {
        return (path, false);
    }
    let dir = format!(
        "{}/.config/xr-desk",
        std::env::var("HOME").unwrap_or_default()
    );
    let store = format!("{dir}/calibration.json");
    if std::path::Path::new(&store).exists() {
        (store, true)
    } else {
        (format!("{dir}/depth-camera.json"), false)
    }
}

fn camera_mtime() -> Option<SystemTime> {
    std::fs::metadata(camera_path().0)
        .and_then(|m| m.modified())
        .ok()
}

/// The camera's place in the head from the file; None when it is missing or unusable.
fn load_camera() -> Option<(Mat3A, Vec3A)> {
    let (path, store) = camera_path();
    let Ok(text) = std::fs::read_to_string(&path) else {
        log::warn!("depth occlusion: no {path}");
        return None;
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else {
        log::warn!("depth occlusion: {path} is not JSON");
        return None;
    };
    let v = if store {
        &v["cameras"]["left"]["pose_in_head"]
    } else {
        &v
    };
    let nums = |k: &str| -> Vec<f32> {
        v[k].as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_f64().map(|x| x as f32))
                    .collect()
            })
            .unwrap_or_default()
    };
    let (r, t) = (nums("r"), nums("t"));
    if r.len() != 9 || t.len() != 3 {
        log::warn!("depth occlusion: {path} needs r (9) and t (3) of the left camera");
        return None;
    }
    log::info!("depth occlusion: camera from {path}, t = {t:?}");
    Some((
        // row-major in the file, glam wants columns
        Mat3A::from_cols_array(&[r[0], r[3], r[6], r[1], r[4], r[7], r[2], r[5], r[8]]),
        Vec3A::new(t[0], t[1], t[2]),
    ))
}

fn f32_at(b: &[u8], o: usize) -> f32 {
    f32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}
fn u64_at(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}

impl State {
    /// Read the camera file again when it changed (it is tuned by hand while XR runs); looked
    /// at twice a second. A file that cannot be used leaves the camera where it was.
    fn refresh_camera(&mut self) {
        let now = monotonic();
        if now - self.cam_checked < CAMERA_CHECK_S {
            return;
        }
        self.cam_checked = now;
        let mtime = camera_mtime();
        if mtime == self.cam_mtime {
            return;
        }
        self.cam_mtime = mtime;
        if let Some((r, t)) = load_camera() {
            self.cam_r = r;
            self.cam_t = t;
        }
    }

    /// Take a new depth map if the tracker wrote one.
    fn refresh(&mut self) {
        self.refresh_camera();
        if self.file.is_none() {
            self.file = File::open("/dev/shm/viture-depth").ok();
        }
        let Some(file) = self.file.as_ref() else {
            return;
        };
        let mut head = [0u8; HEADER];
        if file.read_exact_at(&mut head, 0).is_err() || &head[0..4] != b"VDP1" {
            self.file = None;
            return;
        }
        let count = u64_at(&head, 16);
        if count % 2 == 1 || self.depth.as_ref().is_some_and(|d| d.count == count) {
            return;
        }
        let w = u32::from_le_bytes(head[8..12].try_into().unwrap()) as usize;
        let h = u32::from_le_bytes(head[12..16].try_into().unwrap()) as usize;
        if w == 0 || h == 0 || w * h > 4096 * 4096 {
            return;
        }
        let mut raw = vec![0u8; w * h * 4];
        let mut again = [0u8; 8];
        if file.read_exact_at(&mut raw, HEADER as u64).is_err()
            || file.read_exact_at(&mut again, 16).is_err()
            || u64::from_le_bytes(again) != count
        {
            return; // written meanwhile, next frame
        }
        // header: magic, size, w, h, count u64, stereo seq u64, t_sdk f64, t_mono f64,
        // f, baseline, known, fit error, latency (f32), rotation 9 x f32
        let rot: Vec<f32> = (0..9).map(|i| f32_at(&head, 68 + 4 * i)).collect();
        self.depth = Some(Depth {
            count,
            w,
            h,
            // exposure time of the pair, on the monotonic clock
            t_taken: f64::from_le_bytes(head[32..40].try_into().unwrap()),
            f: f32_at(&head, 48),
            rot: Mat3A::from_cols_array(&[
                rot[0], rot[3], rot[6], rot[1], rot[4], rot[7], rot[2], rot[5], rot[8],
            ]),
            inv: raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        });
    }

    /// The head pose nearest to a time.
    fn pose_at(&self, t: f64) -> Option<Affine3A> {
        self.poses
            .iter()
            .min_by(|a, b| (a.0 - t).abs().total_cmp(&(b.0 - t).abs()))
            .map(|p| p.1)
    }
}

/// Counter of the depth map in use; changes when a screen should be drawn again.
pub fn depth_seq(hmd: &Affine3A) -> u64 {
    let Ok(mut st) = STATE.lock() else {
        return 0;
    };
    let now = monotonic();
    if st.poses.back().is_none_or(|p| now - p.0 > 0.002) {
        st.poses.push_back((now, *hmd));
        while st.poses.len() > 256 {
            st.poses.pop_front();
        }
    }
    st.refresh();
    st.depth.as_ref().map_or(0, |d| d.count)
}

/// The mask of one flat screen as RGBA (black, alpha = hidden), `MASK_W` x `MASK_H`.
///
/// `transform` is the screen's pose in the stage space with its size in the axes' lengths,
/// `aspect` its width over height, `eye` the viewer's eye in the stage space. None when there
/// is no fresh depth map or nothing is hidden.
///
/// Every point of the depth map is put into the room (with the head pose of the moment the
/// map was taken) and carried along the line from the eye through it onto the screen's plane:
/// where that line reaches the point before the screen, the screen is hidden there. So the
/// hole sits where the eye sees the thing, not where the camera, which sits beside the eye,
/// saw it.
pub fn mask(transform: &Affine3A, aspect: f32, eye: Vec3A) -> Option<Vec<u8>> {
    let st = STATE.lock().ok()?;
    let d = st.depth.as_ref()?;
    if monotonic() - d.t_taken > MAX_AGE {
        return None;
    }
    let head = st.pose_at(d.t_taken)?;
    mask_of(d, st.cam_r, st.cam_t, &head, transform, aspect, eye)
}

/// [`mask`] for a given map, camera (head <- camera) and head pose (stage <- head).
fn mask_of(
    d: &Depth,
    cam_r: Mat3A,
    cam_t: Vec3A,
    head: &Affine3A,
    transform: &Affine3A,
    aspect: f32,
    eye: Vec3A,
) -> Option<Vec<u8>> {
    // rectified frame of the map -> left camera -> head as it was (-> stage with `head`)
    let from_rect = cam_r * d.rot.transpose();
    let (sx, sy) = if aspect < 1.0 {
        let major = transform.matrix3.x_axis.length();
        (major * aspect, major)
    } else {
        let major = transform.matrix3.y_axis.length();
        (major, major / aspect)
    };
    if sx < 1e-4 || sy < 1e-4 {
        return None;
    }
    let right = transform.matrix3.x_axis.normalize_or_zero();
    let up = transform.matrix3.y_axis.normalize_or_zero();
    let normal = right.cross(up);
    let centre = transform.translation;
    let to_plane = normal.dot(centre - eye);
    let (cx, cy) = (d.w as f32 * 0.5, d.h as f32 * 0.5);
    let (mw, mh) = (MASK_W as i32, MASK_H as i32);
    let mut out = vec![0u8; (MASK_W * MASK_H * 4) as usize];
    let mut any = false;
    for py in 0..d.h {
        for px in 0..d.w {
            let inv = d.inv[py * d.w + px];
            if inv <= 2e-3 || inv > 20.0 {
                continue; // unknown counts as far away; nearer than 5 cm is noise
            }
            let z = 1.0 / inv;
            let in_rect = Vec3A::new(
                (px as f32 + 0.5 - cx) / d.f * z,
                (py as f32 + 0.5 - cy) / d.f * z,
                z,
            );
            let ray = head.transform_point3a(from_rect * in_rect + cam_t) - eye;
            let along = normal.dot(ray);
            if along.abs() < 1e-6 {
                continue;
            }
            // the screen is reached at `reach` times the way to the point
            let reach = to_plane / along;
            if reach <= 0.0 {
                continue;
            }
            // 1 where the point is clearly nearer than the screen, soft around equal distance
            let hidden = 1.0 / (1.0 + (-(reach - 1.0) / SIGMA * 1.7).exp());
            if hidden <= 0.02 {
                continue;
            }
            let hit = ray * reach + eye - centre;
            let u = hit.dot(right) / sx + 0.5;
            let v = 0.5 - hit.dot(up) / sy;
            if !(-0.1..=1.1).contains(&u) || !(-0.1..=1.1).contains(&v) {
                continue;
            }
            // one pixel of the map covers z / f at the point and `reach` times that on the screen
            let foot = z / d.f * reach * 0.5;
            let (hu, hv) = (
                (foot / sx * MASK_W as f32).min(8.0),
                (foot / sy * MASK_H as f32).min(8.0),
            );
            let (fu, fv) = (u * MASK_W as f32, v * MASK_H as f32);
            let a = (hidden * 255.0) as u8;
            for j in ((fv - hv).floor() as i32).max(0)..=((fv + hv).floor() as i32).min(mh - 1) {
                for i in ((fu - hu).floor() as i32).max(0)..=((fu + hu).floor() as i32).min(mw - 1)
                {
                    let o = ((j * mw + i) * 4 + 3) as usize;
                    if a > out[o] {
                        out[o] = a;
                        any = true;
                    }
                }
            }
        }
    }
    any.then_some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A map that is far everywhere except a square straight ahead of the camera at `near` m.
    fn map_with_square(near: f32) -> Depth {
        let (w, h, f) = (480usize, 288usize, 240.0f32);
        let mut inv = vec![1.0 / 5.0; w * h];
        for y in h / 2 - 12..h / 2 + 12 {
            for x in w / 2 - 12..w / 2 + 12 {
                inv[y * w + x] = 1.0 / near;
            }
        }
        Depth {
            count: 2,
            w,
            h,
            t_taken: 0.0,
            f,
            rot: Mat3A::IDENTITY,
            inv,
        }
    }

    /// Centre of the hidden part of a mask in metres on the screen (x right, y up), and its area
    /// in texels.
    fn hole(mask: &[u8], sx: f32, sy: f32) -> (f32, f32, usize) {
        let (mut su, mut sv, mut n) = (0.0, 0.0, 0usize);
        for j in 0..MASK_H {
            for i in 0..MASK_W {
                if mask[((j * MASK_W + i) * 4 + 3) as usize] > 128 {
                    su += (i as f32 + 0.5) / MASK_W as f32 - 0.5;
                    sv += 0.5 - (j as f32 + 0.5) / MASK_H as f32;
                    n += 1;
                }
            }
        }
        (su / n as f32 * sx, sv / n as f32 * sy, n)
    }

    #[test]
    fn hole_sits_where_the_eye_sees_the_thing() {
        // the head in the stage's origin looking along -z; the camera in the head's origin
        // (OpenCV axes -> OpenXR axes); a screen 1.6 m wide, 1 m ahead, facing the head
        let cam_r = Mat3A::from_cols(Vec3A::X, Vec3A::NEG_Y, Vec3A::NEG_Z);
        let head = Affine3A::IDENTITY;
        let aspect = 16.0 / 9.0;
        let screen = Affine3A::from_scale_rotation_translation(
            glam::Vec3::splat(1.6),
            glam::Quat::IDENTITY,
            glam::Vec3::new(0.0, 0.0, -1.0),
        );
        let d = map_with_square(0.4);
        let (sx, sy) = (1.6, 1.6 / aspect);

        // seen from the camera itself the square at 0.4 m hides the middle of the screen
        let m = mask_of(&d, cam_r, Vec3A::ZERO, &head, &screen, aspect, Vec3A::ZERO).unwrap();
        let (x, y, n) = hole(&m, sx, sy);
        assert!(x.abs() < 0.01 && y.abs() < 0.01, "from the camera: {x} {y}");
        // 24 px of 240 px focal length at 0.4 m = 4 cm, 10 cm on the screen: about 12 x 12 texels
        assert!((100..200).contains(&n), "area {n}");

        // an eye 7 cm to the right and 3 cm above sees it 2.5 times as far the other way
        let eye = Vec3A::new(0.07, 0.03, 0.0);
        let m = mask_of(&d, cam_r, Vec3A::ZERO, &head, &screen, aspect, eye).unwrap();
        let (x, y, _) = hole(&m, sx, sy);
        assert!((x + 0.105).abs() < 0.012, "x {x}");
        assert!((y + 0.045).abs() < 0.012, "y {y}");

        // something behind the screen hides nothing
        let far = map_with_square(1.5);
        assert!(mask_of(&far, cam_r, Vec3A::ZERO, &head, &screen, aspect, eye).is_none());
    }
}
