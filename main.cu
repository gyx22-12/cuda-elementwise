// main.cu —— CUDA 逐元素算子（Add / Sub / Mul）测试
// 方法论：golden 逐元素比对 + 非整除 shape 覆盖矩阵，
//         复现昇腾那次「整数除法丢余数 → 尾元素不被计算」的正确性场景。
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include "elementwise.cuh"

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(_e));                               \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

// ---- golden：CPU 逐元素 ----
template <typename T, typename Op>
static std::vector<T> golden(const std::vector<T>& x, const std::vector<T>& y, Op op) {
    std::vector<T> g(x.size());
    for (size_t i = 0; i < x.size(); ++i) g[i] = op(x[i], y[i]);
    return g;
}

// ---- 校验：float 精确（IEEE 单次 +/-/* 逐位一致），half 用容差 ----
static bool verify(const std::vector<float>& o, const std::vector<float>& g) {
    for (size_t i = 0; i < o.size(); ++i)
        if (o[i] != g[i]) return false;
    return true;
}
static bool verify(const std::vector<half>& o, const std::vector<half>& g) {
    for (size_t i = 0; i < o.size(); ++i) {
        float d = std::fabs(__half2float(o[i]) - __half2float(g[i]));
        float s = std::fmax(1.0f, std::fabs(__half2float(g[i])));
        if (d > 1e-3f * s) return false;
    }
    return true;
}

template <typename T, typename Op>
static bool runCase(int n, int grid, Op op, int version, int threads = 256) {
    std::vector<T> x(n), y(n), z(n);
    for (int i = 0; i < n; ++i) {
        x[i] = T(1.0f + 0.001f * (i % 97));
        y[i] = T(2.0f + 0.0005f * (i % 31));
    }
    std::vector<T> g = golden(x, y, op);

    T *dx = nullptr, *dy = nullptr, *dz = nullptr;
    CUDA_CHECK(cudaMalloc(&dx, n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dy, n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dz, n * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(dx, x.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy, y.data(), n * sizeof(T), cudaMemcpyHostToDevice));

    launchElementwise(dx, dy, dz, n, op, version, grid, threads);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(z.data(), dz, n * sizeof(T), cudaMemcpyDeviceToHost));
    bool ok = verify(z, g);

    cudaFree(dx); cudaFree(dy); cudaFree(dz);
    return ok;
}

int main() {
    struct Case { int n; int grid; const char* desc; };
    const Case cases[] = {
        {921645, 80, "简历原场景：921645/80 余 45"},
        {  8000, 80, "整除"},
        {  8192, 80, "非整除（余 32）"},
        {    50, 80, "totalLength < grid（base=0）"},
        {     1, 80, "单元素"},
    };
    const char* opNames[] = {"add", "sub", "mul"};
    const char* verNames[] = {"block-partition(忠实移植)", "grid-stride", "vec4 向量化"};

    int passed = 0, failed = 0;
    for (int v = 0; v < 3; ++v) {
        std::printf("== %s ==\n", verNames[v]);
        for (const auto& c : cases) {
            for (int opid = 0; opid < 3; ++opid) {
                bool ok;
                if (opid == 0)      ok = runCase<float>(c.n, c.grid, AddOp<float>{}, v);
                else if (opid == 1) ok = runCase<float>(c.n, c.grid, SubOp<float>{}, v);
                else                ok = runCase<float>(c.n, c.grid, MulOp<float>{}, v);
                if (!ok) {
                    std::printf("  [Failed] %-4s %s\n", opNames[opid], c.desc);
                    ++failed;
                } else {
                    ++passed;
                }
            }
        }
    }

    // fp16 版：Add，容差校验
    {
        bool ok = runCase<half>(921645, 80, AddOp<half>{}, 0);
        std::printf("== fp16 ==\n  %s add (921645/80)\n", ok ? "[Success]" : "[Failed]");
        ok ? ++passed : ++failed;
    }

    std::printf("passed=%d failed=%d\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
