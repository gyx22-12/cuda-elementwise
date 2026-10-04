# CUDA 逐元素算子（昇腾 CANN Add/Sub/Mul 的 CUDA 移植）

把华为昇腾训练营里用 AscendC 写的三个逐元素算子（Add / Sub / Mul）用 CUDA 重写一遍，
语义完全一致：**`z[i] = x[i] op y[i]`，支持 fp32 / fp16**。

目的不是「把逐元素写得比 cuBLAS 快」（那是没意义的），而是**打通昇腾和 CUDA 两套编程模型**，
把昇腾侧的 tiling / 双缓冲 / 核间切分，映射到 CUDA 的 grid / block / thread 模型上。

## 昇腾 → CUDA 映射

| 昇腾 AscendC | CUDA |
|---|---|
| `blockDim`（AI Core 数，如 80） | `gridDim.x`（block 数） |
| 核间切分 `base / tail / blockOffset` | `elementwise_block_kernel` 同一套切分 |
| UB 统一缓冲区 + 双缓冲（`BUFFER_NUM=2`） | **无显式双缓冲**：逐元素是 memory-bound，靠足够多的线程 + warp 调度器隐藏访存延迟 |
| Vector 指令（一次搬多个元素） | `float4` 向量化访存（`elementwise_vec4_kernel`） |
| `DataCopy`（GM↔UB 显式搬数） | 自动：全局内存直接读 + L1/L2 缓存 |
| `TilingFunc` / `SetBlockDim` | `<<<grid, block>>>` 启动配置 |

## 三个 kernel 版本

1. **`elementwise_block_kernel`**（忠实移植）：照着昇腾模板 `Init()` 里的 `baseBlockLength /
   tailBlockCount / extra / blockOffset` 切分，余数均匀摊给前几个 block——正是昇腾那次
   「整数除法丢余数 → 尾部 45 个元素不被计算」的 bug 的正确写法。
2. **`elementwise_gridstride_kernel`**（惯用写法）：一个线程跨 grid 步进处理多个元素，天然覆盖任意长度。
3. **`elementwise_vec4_kernel`**（向量化）：一次 `float4` 读 4 个，尾部标量补齐，对应昇腾 Vector 指令。

## 正确性验证

golden 逐元素比对，覆盖非整除 shape 矩阵（`main.cu`）：

| totalLength | grid | 场景 |
|---|---|---|
| 921645 | 80 | 简历原场景（余 45） |
| 8000 | 80 | 整除 |
| 8192 | 80 | 非整除（余 32） |
| 50 | 80 | totalLength < grid（base=0） |
| 1 | 80 | 单元素 |

fp32 做精确比对（IEEE 单次 `+ - *` 逐位一致），fp16 做容差比对。

## 构建 & 运行

```bash
# 方式一：CMake
cmake -B build && cmake --build build && ./build/elementwise

# 方式二：nvcc 直编（Colab 里直接 `bash run.sh`）
nvcc -std=c++17 -arch=all-major -o elementwise main.cu
./elementwise
```

期望输出：三个版本 × 五个 shape × 三个算子全部通过，末尾 `failed=0`。

## 一个真实的调试案例：`__device__` 仿函数被主机调用

首版 `./elementwise` 只打印第一行 `== block-partition ==` 就退出（`exit=1`），且**没有任何 CUDA error**——`CUDA_CHECK` 没触发、`[Failed]` 也没打印。定位过程：

1. **最小二分排除环境**：先写一个单 block/单线程的 `add1` 内核跑通，确认 nvcc + 驱动 + 显卡本身没问题。
2. **逐 case 插桩**：在 `main.cu` 循环里给每个 (version, op, shape) 打 `[vN op n] begin / ok=` 标记，`setvbuf(stdout, _IONBF)` 强制刷新，发现崩在第一个 case 的 `golden()`。
3. **定位根因**：`golden()` 是主机函数，却调用 `AddOp/SubOp/MulOp::operator()`；后者被标成 `__device__`（只能跑在 GPU 上），主机调用是执行空间错误 → 运行时静默崩溃、不产生 CUDA error。

修法：三个仿函数的 `operator()` 从 `__device__ __forceinline__` 改成 `__host__ __device__ __forceinline__`，主机/设备两侧都可内联调用。

修复后在 RTX 3080 Ti（sm_86）上 46 例全过（`passed=46 failed=0`）。
