#version 460
#extension GL_GOOGLE_include_directive : require              // For #include
#extension GL_EXT_scalar_block_layout : require               // For scalar layout
#extension GL_EXT_shader_explicit_arithmetic_types : require  // For uint64_t, ...
#extension GL_EXT_buffer_reference2 : require                 // For buffer reference

#include "shader_io.h"


layout(location = LVPosition) in vec3 inPosition;
layout(location = LVColor) in vec3 inColor;
layout(location = LVTexCoord) in vec2 inUv;

layout(location = 0) out vec3 outColor;
layout(location = 1) out vec2 outUv;

// Push constants: carries the scene buffer address and per-draw color.
// Backed by a real VkPipelineLayout with a VkPushConstantRange (shared with the fragment shader).
layout(push_constant, scalar) uniform GraphicsPushData_
{
  GraphicsPushData pushData;
};

// SceneInfo lives in a GPU buffer; we access it via buffer device address from push data.
layout(buffer_reference, scalar) readonly buffer SceneInfoRef { SceneInfo sceneInfo; };


void main()
{
  // Access the scene info buffer via its device address
  SceneInfoRef scene = SceneInfoRef(pushData.sceneInfoAddress);

  vec3 pos = inPosition;

  // Adjust aspect ratio using resolution from the scene buffer
  float aspectRatio = scene.sceneInfo.resolution.y / scene.sceneInfo.resolution.x;
  //pos.x *= aspectRatio;
  // Set the position in clip space
  gl_Position = vec4(pos, 1.0);
  // Pass the color and uv
  outColor = inColor;
  outUv    = inUv;
}
