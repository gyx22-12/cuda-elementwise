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
