#version 310 es
precision highp float;
precision highp int;

// Relief panel, vertex stage: reads the header row of the depth strip (bottom row of the window
// picture) once per vertex and hands it on flat. Layout of the strip: onshape-xr, depth.js.

layout (location = 0) in vec2 in_pos;
layout (location = 1) in vec2 in_uv;
layout (location = 0) out vec2 out_uv;
layout (location = 1) flat out vec4 h_strip;   // strip w, h (pixels); header x, y in window pixels
layout (location = 2) flat out vec4 h_canvas;  // canvas x, y (top-left origin), w, h in window pixels
layout (location = 3) flat out vec4 h_proj;    // P[0], P[5], P[10], P[14]
layout (location = 4) flat out float h_valid;

layout (set = 0, binding = 0) uniform highp sampler2D win;

// The header: the last row of the page, starting at its left edge. A browser window may carry
// margins around the page (shadows), so the header is searched near the bottom left corner.
ivec2 origin = ivec2(0);

uint byte_at(int k) {
    return uint(texelFetch(win, origin + ivec2(k / 3, 0), 0)[k % 3] * 255.0 + 0.5);
}

float hf(int n) {
    int b = 6 + 4 * n;
    return uintBitsToFloat(byte_at(b) | (byte_at(b + 1) << 8)
        | (byte_at(b + 2) << 16) | (byte_at(b + 3) << 24));
}

bool magic_at(ivec2 p) {
    ivec3 a = ivec3(texelFetch(win, p, 0).rgb * 255.0 + 0.5);
    if (a != ivec3(79, 88, 82)) {
        return false;
    }
    ivec3 b = ivec3(texelFetch(win, p + ivec2(1, 0), 0).rgb * 255.0 + 0.5);
    return b == ivec3(68, 48, 49);
}

void main() {
    out_uv = in_uv;
    gl_Position = vec4(in_pos * 2. - 1., 0., 1.);

    ivec2 ws = textureSize(win, 0);
    bool ok = false;
    for (int dy = 0; dy < 96 && !ok; dy++) {
        for (int dx = 0; dx < 96; dx++) {
            ivec2 p = ivec2(dx, ws.y - 1 - dy);
            if (magic_at(p)) {
                origin = p;
                ok = true;
                break;
            }
        }
    }
    h_valid = ok ? 1.0 : 0.0;
    h_strip = vec4(hf(1), hf(2), float(origin.x), float(origin.y));
    float cw = hf(5);
    float ch = hf(6);
    h_canvas = vec4(float(origin.x) + hf(3), float(origin.y + 1) - hf(4) - ch, cw, ch);
    h_proj = vec4(hf(7), hf(8), hf(9), hf(10));
}
