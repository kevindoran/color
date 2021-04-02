export default `#version 300 es
precision mediump float;

uniform vec2 u_resolution;
uniform vec3 u_circle_rgb;
uniform vec3 u_bg_rgb;
out vec4 out_color;

bool in_circle(vec2 center, float radius, vec2 point) {
  return step(distance(point, center), radius) > 0.0;
}

void main(void) {
  vec4 color = vec4(1.0, 1.0, 1.0, 1.0);
  vec2 st = gl_FragCoord.xy / u_resolution.xy;
  float x_factor = u_resolution.x / u_resolution.y;
  st.x *= x_factor;
  vec2 mid =  vec2(0.5*x_factor, 0.5);
  if (in_circle(mid, 0.3, st)) {
    color.xyz = u_circle_rgb;
  }
  else {
    color.xyz = u_bg_rgb;
  }
  out_color = color;
}
`;
