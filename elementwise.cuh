// elementwise.cuh
// 昇腾 AscendC 逐元素算子（Add / Sub / Mul）的 CUDA 移植。
//
// 语义：z[i] = x[i] op y[i]，逐元素；数据类型 float (DT_FLOAT) / half (DT_FLOAT16)。
//
// ── 昇腾 → CUDA 映射 ─────────────────────────────────────────────
//   AscendC blockDim（AI Core 数）  →  CUDA gridDim.x（block 数）
//   核间切分 base/tail/blockOffset   →  elementwise_block_kernel 同一套切分
//   UB 双缓冲（BUFFER_NUM=2）        →  CUDA 无需显式双缓冲：逐元素是 memory-bound，
//                                         靠足够多的线程 + warp 调度器把访存延迟藏掉
//   Vector 指令（一次搬多个元素）     →  float4 向量化访存（elementwise_vec4_kernel）
// ────────────────────────────────────────────────────────────────

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>

// ---- 三个算子：模板仿函数，float / half 各自用原生 + - * -------------------
template <typename T>
struct AddOp {
    __device__ __forceinline__ T operator()(T a, T b) const { return a + b; }
};
template <typename T>
struct SubOp {
    __device__ __forceinline__ T operator()(T a, T b) const { return a - b; }
};
template <typename T>
struct MulOp {
    __device__ __forceinline__ T operator()(T a, T b) const { return a * b; }
};

// ---- 版本一：忠实移植昇腾模板的核间切分 --------------------------------------
// 对应 add_custom_template.cpp 里 Init() 的 baseBlockLength / tailBlockCount /
// extra / blockOffset：把 totalLength 均分给 gridDim.x 个 block，余数 tail 均匀
// 摊给前 tail 个 block（每个多算 1 个元素），保证不丢尾、不越界。
template <typename T, typename Op>
__global__ void elementwise_block_kernel(const T* x, const T* y, T* z, int totalLength, Op op) {
    const int numBlocks = gridDim.x;
    const int blockId   = blockIdx.x;
    const int base      = totalLength / numBlocks;
    const int tail      = totalLength % numBlocks;
    const int extra     = blockId < tail ? 1 : 0;
    const int blockLength = base + extra;
    const int offset      = blockId * base + (blockId < tail ? blockId : tail);

    for (int i = threadIdx.x; i < blockLength; i += blockDim.x) {
        z[offset + i] = op(x[offset + i], y[offset + i]);
    }
}

// ---- 版本二：grid-stride loop（CUDA 惯用写法）------------------------------
// 一个线程处理多个元素、跨 grid 步进，天然覆盖任意长度，不用手写尾块。
template <typename T, typename Op>
__global__ void elementwise_gridstride_kernel(const T* x, const T* y, T* z, int n, Op op) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        z[i] = op(x[i], y[i]);
    }
}

// ---- 版本三：float4 向量化（对应昇腾 Vector 指令）---------------------------
// 一次加载 4 个 float，减少访存指令数；尾部 <4 个元素用标量补齐。仅 float 可用。
template <typename Op>
__global__ void elementwise_vec4_kernel(const float* x, const float* y, float* z, int n, Op op) {
    const int n4 = n / 4;  // 完整 float4 的个数
    const float4* x4 = reinterpret_cast<const float4*>(x);
    const float4* y4 = reinterpret_cast<const float4*>(y);
    float4* z4 = reinterpret_cast<float4*>(z);

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
        float4 a = x4[i];
        float4 b = y4[i];
        z4[i] = make_float4(op(a.x, b.x), op(a.y, b.y), op(a.z, b.z), op(a.w, b.w));
    }
    // 尾部（<4 个）标量补齐
    for (int i = n4 * 4 + blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        z[i] = op(x[i], y[i]);
    }
}

// ---- 启动辅助：按 version 选 kernel -----------------------------------------
//   version 0 = block-partition（忠实移植版，调用方显式给 grid，对应 blockDim=80）
//   version 1 = grid-stride
//   version 2 = float4 向量化（仅 float；half 自动回退到 grid-stride）
template <typename T, typename Op>
void launchElementwise(const T* x, const T* y, T* z, int n, Op op, int version, int grid = 0, int threads = 256) {
    if (version == 0) {
        int g = grid > 0 ? grid : ((n + threads - 1) / threads);
        elementwise_block_kernel<<<g, threads>>>(x, y, z, n, op);
    } else if (version == 1) {
        int blocks = (n + threads - 1) / threads;
        if (blocks > 1024) blocks = 1024;
        elementwise_gridstride_kernel<<<blocks, threads>>>(x, y, z, n, op);
    } else if constexpr (std::is_same_v<T, float>) {
        int blocks = (n + threads - 1) / threads;
        if (blocks > 1024) blocks = 1024;
        elementwise_vec4_kernel<<<blocks, threads>>>(x, y, z, n, op);
    } else {
        // half 无 float4 版，回退到 grid-stride
        int blocks = (n + threads - 1) / threads;
        if (blocks > 1024) blocks = 1024;
        elementwise_gridstride_kernel<<<blocks, threads>>>(x, y, z, n, op);
    }
}
