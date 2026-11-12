#version 310 es
precision highp float;
precision highp int;

// Relief panel, fragment stage: the 3D view of a page as a surface in space. The window picture
// holds the view's colour in the canvas rectangle and its depth in a strip at the bottom left.
// For every pixel of the panel a ray from the eye is marched through the view's box until it
// meets the surface.
//
// Panel space: the panel spans x in [-0.5, 0.5], y in [-0.5/aq, 0.5/aq], z = 0, +z towards the
// viewer. The box sits centred on the panel.

layout (location = 0) in vec2 in_uv;
layout (location = 1) flat in vec4 h_strip;
layout (location = 2) flat in vec4 h_canvas;
layout (location = 3) flat in vec4 h_proj;
layout (location = 4) flat in float h_valid;
layout (location = 0) out vec4 out_color;

layout (set = 0, binding = 0) uniform highp sampler2D win;
layout (set = 1, binding = 0) uniform Params {
    vec4 eye;   // xyz: eye in panel space
    vec4 cfg;   // x: box size relative to the panel, y: panel aspect (w/h), z: 1 = show the frame
};

const int STEPS = 72;
const int REFINE = 5;

// Surface height (panel z) at canvas uv; x = height, y = 1 if there is a surface.
vec2 surface(vec2 cuv, float hz) {
    ivec2 sp = ivec2(clamp(cuv, vec2(0.0), vec2(0.99999)) * h_strip.xy);
    ivec2 wp = ivec2(int(h_strip.z) + sp.x, int(h_strip.w) - int(h_strip.y) + sp.y);
    vec3 t = texelFetch(win, wp, 0).rgb;
    float d = (t.r * 255.0 * 256.0 + t.g * 255.0) / 65535.0;
    return vec2((1.0 - 2.0 * d) * hz, step(0.5, t.b));
}

void main() {
    float aq = cfg.y;
    vec2 edge = min(in_uv, 1.0 - in_uv) * vec2(1.0, 1.0 / aq);
    vec4 frame = (cfg.z > 0.5 && min(edge.x, edge.y) < 0.004) ? vec4(0.2, 0.6, 1.0, 0.8) : vec4(0.0);
    out_color = frame;
    if (h_valid < 0.5) {
        return;
    }

    float a = h_canvas.z / h_canvas.w;
    float m = cfg.x * min(1.0, a / aq);
    vec3 half_box = vec3(0.5 * m, 0.5 * m / a, 0.5 * m * h_proj.x / abs(h_proj.z));
    float hz = half_box.z;
    vec3 q = vec3(in_uv.x - 0.5, (0.5 - in_uv.y) / aq, 0.0);
    vec3 dir = normalize(q - eye.xyz);

    vec3 inv = 1.0 / dir;
    vec3 ta = (-half_box - eye.xyz) * inv;
    vec3 tb = (half_box - eye.xyz) * inv;
    vec3 tmin = min(ta, tb);
    vec3 tmax = max(ta, tb);
    float t0 = max(max(tmin.x, tmin.y), max(tmin.z, 0.0));
    float t1 = min(min(tmax.x, tmax.y), tmax.z);
    if (t1 <= t0) {
        return;
    }

    float dt = (t1 - t0) / float(STEPS);
    float thick = max(abs(dir.z) * dt * 2.5, hz * 0.03);
    float tp = t0;
    for (int i = 1; i <= STEPS; i++) {
        float t = t0 + dt * float(i);
        vec3 p = eye.xyz + dir * t;
        vec2 s = surface(vec2(p.x / m + 0.5, 0.5 - p.y * a / m), hz);
        if (s.y > 0.5 && p.z <= s.x && p.z >= s.x - thick) {
            float lo = tp;
            float hi = t;
            for (int k = 0; k < REFINE; k++) {
                float mid = 0.5 * (lo + hi);
                vec3 pm = eye.xyz + dir * mid;
                vec2 sm = surface(vec2(pm.x / m + 0.5, 0.5 - pm.y * a / m), hz);
                if (sm.y > 0.5 && pm.z <= sm.x) {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            vec3 ph = eye.xyz + dir * hi;
            vec2 huv = vec2(ph.x / m + 0.5, 0.5 - ph.y * a / m);
            vec2 px = h_canvas.xy + clamp(huv, vec2(0.0), vec2(1.0)) * h_canvas.zw;
            out_color = vec4(textureLod(win, px / vec2(textureSize(win, 0)), 0.0).rgb, 1.0);
            return;
        }
        tp = t;
    }
}
