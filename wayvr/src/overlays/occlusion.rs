//! Real things in front of a screen hide it: a mask from the glasses' depth map.
//!
//! A tracker outside this program publishes a metric depth map of the left tracking camera in
//! shared memory (`/dev/shm/viture-depth`, block `VDP1`). For a screen, [`mask`] walks a coarse
//! grid over its quad, puts every grid point into that camera as it stood when the map was
//! taken, and compares distances: where the room is nearer than the screen, the mask is opaque
//! black, which is see-through on additive glasses. The screen backend draws the mask over its
//! picture with the cursor pipeline.
//!
//! Off unless `WAYVR_DEPTH_OCCLUSION=1`. The camera's place in the head is read from
//! `WAYVR_DEPTH_CAMERA` (a JSON file, default `~/.config/xr-desk/depth-camera.json`):
//! `{"r": [9 numbers, row-major], "t": [3]}` with `p_head = r * p_camera + t`, the camera in
//! OpenCV axes (x right, y down, z forward), the head in OpenXR view axes. Without the file the
//! camera sits in the head's origin looking forward. The file is read again when it changes.

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

fn camera_path() -> String {
    std::env::var("WAYVR_DEPTH_CAMERA").unwrap_or_else(|_| {
        format!(
            "{}/.config/xr-desk/depth-camera.json",
            std::env::var("HOME").unwrap_or_default()
        )
    })
}

fn camera_mtime() -> Option<SystemTime> {
    std::fs::metadata(camera_path())
        .and_then(|m| m.modified())
        .ok()
}

/// The camera's place in the head from the file; None when it is missing or unusable.
fn load_camera() -> Option<(Mat3A, Vec3A)> {
    let path = camera_path();
    let Ok(text) = std::fs::read_to_string(&path) else {
        log::warn!("depth occlusion: no {path}");
        return None;
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else {
        log::warn!("depth occlusion: {path} is not JSON");
        return None;
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
        log::warn!("depth occlusion: {path} needs r (9) and t (3)");
        return None;
    }
    log::info!("depth occlusion: camera from {path}");
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
/// `aspect` its width over height. None when there is no fresh depth map or nothing is hidden.
pub fn mask(transform: &Affine3A, aspect: f32) -> Option<Vec<u8>> {
    let st = STATE.lock().ok()?;
    let d = st.depth.as_ref()?;
    if monotonic() - d.t_taken > MAX_AGE {
        return None;
    }
    let head = st.pose_at(d.t_taken)?;
    // stage -> head as it was -> left camera -> rectified frame of the map
    let to_head = head.inverse();
    let to_rect = d.rot * st.cam_r.transpose();
    let (sx, sy) = if aspect < 1.0 {
        let major = transform.matrix3.x_axis.length();
        (major * aspect, major)
    } else {
        let major = transform.matrix3.y_axis.length();
        (major, major / aspect)
    };
    let right = transform.matrix3.x_axis.normalize_or_zero() * sx;
    let up = transform.matrix3.y_axis.normalize_or_zero() * sy;
    let (cx, cy) = (d.w as f32 * 0.5, d.h as f32 * 0.5);
    let mut out = vec![0u8; (MASK_W * MASK_H * 4) as usize];
    let mut any = false;
    for j in 0..MASK_H {
        let v = (j as f32 + 0.5) / MASK_H as f32;
        for i in 0..MASK_W {
            let u = (i as f32 + 0.5) / MASK_W as f32;
            let world = transform.translation + right * (u - 0.5) + up * (0.5 - v);
            let p = to_rect * (to_head.transform_point3a(world) - st.cam_t);
            if p.z < 0.05 {
                continue;
            }
            let (px, py) = (d.f * p.x / p.z + cx, d.f * p.y / p.z + cy);
            if px < 0.0 || py < 0.0 || px >= d.w as f32 || py >= d.h as f32 {
                continue;
            }
            let inv = d.inv[py as usize * d.w + px as usize];
            if inv <= 2e-3 {
                continue; // unknown counts as far away
            }
            let room = 1.0 / inv;
            // 1 where the room is clearly nearer than the screen, soft around equal distance
            let hidden = 1.0 / (1.0 + (-(p.z - room) / (SIGMA * room) * 1.7).exp());
            if hidden > 0.02 {
                out[((j * MASK_W + i) * 4 + 3) as usize] = (hidden * 255.0) as u8;
                any = true;
            }
        }
    }
    any.then_some(out)
}
