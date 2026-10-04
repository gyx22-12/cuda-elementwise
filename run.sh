#!/usr/bin/env bash
# 无 CMake 时的直连编译运行（Colab 里直接 `bash run.sh` 即可）。
set -e
nvcc -std=c++17 -arch=all-major -o elementwise main.cu
./elementwise
