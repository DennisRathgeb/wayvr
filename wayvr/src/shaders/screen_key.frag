#version 310 es
precision highp float;

// Screen picture with black as see-through: on additive glasses black is invisible anyway, and
// this lets the layers of other programs (drawn below the overlay) show where a screen is black.

layout (location = 0) in vec2 in_uv;
layout (location = 0) out vec4 out_color;

layout (set = 0, binding = 0) uniform sampler2D in_texture;

void main()
{
    vec3 c = texture(in_texture, in_uv).rgb;
    float a = smoothstep(0.0, 0.04, max(c.r, max(c.g, c.b)));
    out_color = vec4(c * a, a);
}
