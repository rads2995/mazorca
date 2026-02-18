#!/usr/bin/env bash

git clone https://github.com/intel/llvm -b v6.3.0 --depth=1
cmake -S ./llvm/unified-runtime -B ./llvm/build \
    -DUR_BUILD_EXAMPLES=OFF \
    -DUR_BUILD_TESTS=OFF \
    -DUR_BUILD_TOOLS=OFF \
    -DUR_BUILD_ADAPTER_L0=OFF \
    -DUR_BUILD_ADAPTER_OPENCL=ON \
    -DUR_BUILD_ADAPTER_CUDA=OFF \
    -DUR_BUILD_ADAPTER_HIP=ON \
    -DUR_BUILD_ADAPTER_NATIVE_CPU=OFF \
    -DUR_HIP_PLATFORM=AMD \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="./llvm/build/install"
cmake --build ./llvm/build -j6
cmake --install ./llvm/build

# Intel's oneAPI DPC++/C++ Compiler for SYCL support
source /opt/intel/oneapi/setvars.sh
export PATH="$PATH:/opt/intel/oneapi/compiler/latest/bin/compiler/:/opt/intel/oneapi/debugger/latest/bin"

# Add Unified Runtime's adapters and prepend so that our libur_loader.so is used
export LD_LIBRARY_PATH=$(pwd)/llvm/build/install/lib:$LD_LIBRARY_PATH
