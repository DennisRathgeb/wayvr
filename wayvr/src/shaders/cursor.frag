#version 310 es
precision highp float;

// The desktop cursor over a mirrored screen: like screen.frag, but the image's alpha is kept.

layout (location = 0) in vec2 in_uv;
layout (location = 0) out vec4 out_color;

layout (set = 0, binding = 0) uniform sampler2D in_texture;

void main()
{
    out_color = texture(in_texture, in_uv);
}
