#version 460

#extension GL_GOOGLE_include_directive : require              // For #include
#extension GL_EXT_scalar_block_layout : require               // For scalar layout
#extension GL_EXT_shader_explicit_arithmetic_types : require  // For uint64_t, ...
#extension GL_EXT_buffer_reference2 : require                 // For buffer reference
#extension GL_EXT_nonuniform_qualifier : require              // For non-uniform indexing of the texture array
#extension GL_EXT_descriptor_heap : enable                    // For bindless descriptor heap access (VK_EXT_descriptor_heap)

#include "shader_io.h"

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 inUv;

layout(location = 0) out vec4 outColor;

// Descriptor heap: textures and samplers are stored in GPU heap buffers (not descriptor sets).
// The shader accesses them by index. Heap index 0 holds our linear sampler (written in createDescriptorHeap).
// Image heap indices 0..N correspond to the loaded textures (also written in createDescriptorHeap).
layout(descriptor_heap) uniform texture2D heapTextures[];  // Image descriptors from the resource heap
layout(descriptor_heap) uniform sampler   heapSamplers[];  // Sampler descriptors from the sampler heap

// Push data (vkCmdPushDataEXT): only carries the scene buffer address and per-draw color.
// With descriptor heap the pipeline layout is VK_NULL_HANDLE, so push constants/descriptors can't be used.
layout(push_constant, scalar) uniform GraphicsPushData_
{
  GraphicsPushData pushData;
};

// Buffer references: GPU buffers accessed by their device address (buffer device address / BDA).
// SceneInfo is a GPU buffer updated once per frame; its address comes from push data.
// Datas is the points buffer; its address comes from SceneInfo.
layout(buffer_reference, scalar) readonly buffer SceneInfoRef { SceneInfo sceneInfo; };
layout(buffer_reference, scalar) readonly buffer Datas { vec2 _[]; };

// Specialization constant
layout(constant_id = 0) const bool useTexture = false;


void reference()
{
  // Access the scene info buffer via its device address (updated once per frame on the CPU side)
  SceneInfoRef scene = SceneInfoRef(pushData.sceneInfoAddress);

  // Compute the normalized fragment position and center it at (0, 0)
  vec2 fragPos = (gl_FragCoord.xy / scene.sceneInfo.resolution) * 2.0 - 1.0;

  // Access the points data buffer via buffer device address (stored inside SceneInfo)
  Datas datas = Datas(scene.sceneInfo.dataBufferAddress);

  // Loop over points in the data buffer
  // Compute the distance between the fragment and each point (uniform screen space, not moving with triangle)
  float minDist = 1e10;
  for(int i = 0; i < scene.sceneInfo.numData; i++)
  {
    vec2  pnt  = datas._[i];
    float dist = distance(fragPos, pnt);
    minDist    = min(minDist, dist);
  }

  // Create a smooth transition around the points' boundaries (anti-aliasing effect)
  float radius     = 0.02;
  float edgeSmooth = 0.01;  // Smooth the edge
  float alpha      = 1.0 - smoothstep(radius, radius - edgeSmooth, minDist);

  vec4 pointColor = vec4(scene.sceneInfo.animValue * pushData.color, 1.0);  // points flashing using the per-draw color
  vec4 triangleColor = vec4(fragColor, 1.0);                                // Interpolated color from the vertex shader

  // Sample texture using descriptor heap: combine image and sampler by their heap indices.
  // heapTextures[texId] picks the image from the resource heap; heapSamplers[0] picks the linear sampler.
  // nonuniformEXT is required because the index may vary across invocations.
  if(useTexture)
    triangleColor *= texture(sampler2D(heapTextures[nonuniformEXT(scene.sceneInfo.texId)], heapSamplers[0]), inUv);

  // Blend the point with the background based on the minimum distance
  //outColor = mix(pointColor, triangleColor, alpha);
  outColor = triangleColor;
}

void main()
{
    // Access the scene info buffer via its device address to retrieve the texture index
    SceneInfoRef scene = SceneInfoRef(pushData.sceneInfoAddress);

    uint x = uint(gl_FragCoord.x);
    uint y = uint(gl_FragCoord.y);

    vec2 uv = vec2(0.0, 0.0);
    outColor = vec4(0.0, 1.0, 0.0, 1.0);

    if (((x & 1) == 0) && ((y & 1) == 0))
    {
      vec2 uv = inUv * 0.5f;

      // textureQueryLod returns (accessed mip level, computed lod)
      vec2 lodInfo = textureQueryLod(sampler2D(heapTextures[nonuniformEXT(scene.sceneInfo.texId)], heapSamplers[0]), uv);

      float mipLevel   = lodInfo.x;
      float computedLod = lodInfo.y;

      // Normalize for visualization.
      // Change maxMip according to your texture's mip count.
      float maxMip = 8.0;
      float t = clamp(computedLod / maxMip, 0.0, 1.0);

      // Simple heatmap:
      // blue = low LOD, red = high LOD
      vec3 color = mix(vec3(0.0, 0.2, 1.0), vec3(1.0, 0.0, 0.0), t);

      outColor = vec4(color, 1.0);
    }
}
