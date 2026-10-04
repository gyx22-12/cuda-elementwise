cmake_minimum_required(VERSION 3.18)
project(cuda_elementwise LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# all-major：为所有主流架构生成 PTX，运行时 JIT，免去逐卡指定（Colab T4=75 / A100=80 / H100=90）。
set(CMAKE_CUDA_ARCHITECTURES all-major)

add_executable(elementwise main.cu)
