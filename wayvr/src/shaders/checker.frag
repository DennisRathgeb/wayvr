#version 450
// A chessboard like the printed ChArUco board: squares.x by squares.y cells, white margin.
layout (location = 0) in vec2 in_uv;
layout (location = 0) out vec4 out_color;
layout (set = 0, binding = 0) uniform Params {
    vec4 squares;   // x: columns, y: rows, z: margin as a fraction of the width, w: unused
    vec4 color;     // tint and alpha of the dark squares
} params;

void main() {
    vec2 uv = in_uv;
    float m = params.squares.z;
    vec2 inner = (uv - vec2(m)) / (1.0 - 2.0 * m);
    if (inner.x < 0.0 || inner.x > 1.0 || inner.y < 0.0 || inner.y > 1.0) {
        out_color = vec4(1.0, 1.0, 1.0, 0.35);   // the paper margin, faint
        return;
    }
    ivec2 cell = ivec2(floor(inner * params.squares.xy));
    bool dark = ((cell.x + cell.y) & 1) == 0;
    // thin bright lines on the square edges help the eye align it with the print
    vec2 f = fract(inner * params.squares.xy);
    float edge = min(min(f.x, 1.0 - f.x), min(f.y, 1.0 - f.y));
    if (edge < 0.04) {
        out_color = vec4(0.2, 1.0, 0.2, 1.0);
        return;
    }
    out_color = dark ? params.color : vec4(1.0, 1.0, 1.0, 0.25);
}
