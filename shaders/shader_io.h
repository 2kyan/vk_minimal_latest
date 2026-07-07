
#ifndef HOST_DEVICE_H
#define HOST_DEVICE_H

#ifdef __SLANG__
typealias vec2 = float2;
typealias vec3 = float3;
typealias vec4 = float4;
#define STATIC_CONST static const
#else
#define STATIC_CONST const
#endif

// Vertex layout
STATIC_CONST int LVPosition = 0;
STATIC_CONST int LVColor    = 1;
STATIC_CONST int LVTexCoord = 2;

// Scene information, stored in a GPU buffer and updated once per frame via vkCmdUpdateBuffer.
// The shader accesses it through a buffer device address (buffer reference), not a descriptor set.
struct SceneInfo
{
  uint64_t dataBufferAddress;  // Buffer device address of the points data buffer
  vec2     resolution;         // Viewport resolution
  float    animValue;          // Animation value (sine wave)
  int      numData;            // Number of points in the data buffer
  int      texId;              // Which texture to sample (index into the descriptor set texture array)
};

// Push constants for the graphics pipeline.
// Carried via a real VkPipelineLayout push constant range covering VS + FS stages.
// Holds the minimum needed per draw call:
// the address of the scene buffer (updated once per frame) and the per-draw color.
struct GraphicsPushData
{
  uint64_t sceneInfoAddress;  // Buffer device address of the SceneInfo GPU buffer
  vec3     color;             // Per-draw triangle color (changes between draw calls)
};

struct PushConstantCompute
{
  uint64_t bufferAddress;  // Buffer device address of the vertex buffer
  float    rotationAngle;
  int      numVertex;
};

struct Vertex
{
  vec3 position;
  vec3 color;
  vec2 texCoord;
};


#endif  // HOST_DEVICE_H
