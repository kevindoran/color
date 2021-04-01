import fsSource from './fragShader.js';


main();
function main() {
  const canvas = document.querySelector('#glcanvas');
  const gl = canvas.getContext('webgl2');

  resize(canvas);
  // If we don't have a GL context, give up now

  if (!gl) {
    alert('Unable to initialize WebGL. Your browser or machine may not support it.');
    return;
  }

  // Vertex shader program

  const vsSource = `#version 300 es
  in vec4 a_position;
  void main(void) {
    gl_Position = a_position;
  }`;

  const prg = initShaderProgram(gl, vsSource, fsSource);
  drawScene(gl, prg);
}


function resize(canvas) {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
}


function drawScene(gl, prg) {
  gl.clearColor(0.0, 1.0, 0.0, 1.0);  // Clear to black, fully opaque
  gl.clearDepth(1.0);                 // Clear everything
  gl.enable(gl.DEPTH_TEST);           // Enable depth testing
  gl.depthFunc(gl.LEQUAL);            // Near things obscure far things

  // Clear the canvas before we start drawing on it.
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  gl.useProgram(prg);

  const positionAttributeLocation = gl.getAttribLocation(prg, 'a_position');
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const positionBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    -1, -1, 
    1, -1,
    -1, 1,
    -1, 1, 
    1, -1,
    1, 1]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(positionAttributeLocation);
  gl.vertexAttribPointer(
    positionAttributeLocation, 
    2, 
    gl.FLOAT,
    false,
    0, 
    0);
  const circle_color_location = gl.getUniformLocation(prg, 'u_circle_color');
  const bg_color_location = gl.getUniformLocation(prg, 'u_bg_color');
  const position_location = gl.getUniformLocation(prg, 'u_position');
  const resolution_location = gl.getUniformLocation(prg, 'u_resolution');
  gl.uniform3f(circle_color_location, 1.0, 0.0, 0.0);
  gl.uniform3f(bg_color_location, 0.3, 0.3, 0.3);
  gl.uniform2f(resolution_location, gl.drawingBufferWidth, gl.drawingBufferHeight);
  gl.uniform2f(position_location, 0.5, 0.5);

  gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
  gl.bindVertexArray(vao);
  {
    const offset = 0;
    const vertexCount = 6;
    gl.drawArrays(gl.TRIANGLES, offset, vertexCount);
  }
}


function initShaderProgram(gl, vsSource, fsSource) {
  const vertexShader = loadShader(gl, gl.VERTEX_SHADER, vsSource);
  const fragmentShader = loadShader(gl, gl.FRAGMENT_SHADER, fsSource);
  const shaderProgram = gl.createProgram();
  gl.attachShader(shaderProgram, vertexShader);
  gl.attachShader(shaderProgram, fragmentShader);
  gl.linkProgram(shaderProgram);
  if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) {
    alert('Unable to initialize the shader program: ' + gl.getProgramInfoLog(shaderProgram));
    return null;
  }
  return shaderProgram;
}


function loadShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    alert('An error occurred compiling the shaders: ' + gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

