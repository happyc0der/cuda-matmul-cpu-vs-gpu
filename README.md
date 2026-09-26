# cuda-matmul-cpu-vs-gpu

My first CUDA experiments (March 2025): a "hello world" kernel, then a naive
matrix-multiplication kernel timed against a plain single-threaded C++ loop on
1024 x 1024 matrices.

These are learning exercises, not an optimised GEMM. There is no shared-memory
tiling, cuBLAS or error checking. Each is a small next step you could add.

## Files

| File | What it does |
|------|--------------|
| `hello.cu` | Launches one GPU thread (`<<<1, 1>>>`) that prints `Hello, World from CUDA!` with device-side `printf`. |
| `matrix_mul.cu` | Naive CUDA matrix multiply, `C = A x B`, for 1024 x 1024 `float` matrices. Uses 32 x 32 thread blocks and prints two result entries to check them. No timing. |
| `matrix_mul.cpp` | The same multiply as a CPU triple loop (single thread), timed with `std::chrono`. Same sizes and inputs as the CUDA version. |
| `matrix_mul_compare.cu` | Runs the CPU loop and the GPU kernel (16 x 16 blocks) in one program. It times both, prints the speedup and checks every element of the two results. |
| `test.jl` | A one-line Julia "hello world" used to check the Julia install. It is unrelated to CUDA. |
| `build.ps1` | Windows build script (nvcc + MSVC). |
| `Makefile` | Linux/WSL build (nvcc + g++). |

## The technique

**Naive kernel (one thread per output element).** The grid is 2-D. Each
thread works out its `(row, col)` from `blockIdx`, `blockDim` and
`threadIdx`, then computes the dot product of row `row` of `A` and column
`col` of `B`:

```cuda
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
if (row < width && col < width) {
    float sum = 0.0f;
    for (int k = 0; k < width; k++)
        sum += A[row * width + k] * B[k * width + col];
    C[row * width + col] = sum;
}
```

The grid size is rounded up (`(width + block - 1) / block`), so the bounds
check covers sizes that are not a multiple of the block size. Every thread
reads its operands straight from global memory, with no reuse through
shared memory. That makes it the baseline that tiled kernels improve on.

**Host flow.** The program allocates device buffers with `cudaMalloc`, copies
`A` and `B` to the GPU with `cudaMemcpy`, launches the kernel and calls
`cudaDeviceSynchronize`. It then copies `C` back and frees the buffers.

**CPU baseline.** The same `row / col / k` triple loop runs on one thread
with no blocking or SIMD intrinsics.

**Inputs and checks.** Every entry of `A` is `1.0` and every entry of `B` is
`2.0`, so every entry of `C` should equal `2 * width = 2048`.

### How to read the timings

- In `matrix_mul_compare.cu`, the **GPU time is measured end to end on the
  host**. It includes `cudaMalloc`, both host-to-device copies, the kernel,
  the device-to-host copy and `cudaFree`. It also includes the one-off CUDA
  context creation, because the first CUDA call in the process happens inside
  the timed region. The kernel alone is much faster than this number. Use CUDA
  events or Nsight Systems to time it on its own.
- The CPU time is one run of an unoptimised single-threaded loop. It depends a
  lot on the compiler and flags (see the results below).
- The programs don't check CUDA API or launch errors. If there's no GPU or the
  launch fails, the GPU result is left uninitialised and the comparison
  prints `Results don't match`.

## Building

You need an NVIDIA GPU, the CUDA Toolkit (`nvcc`) and a host C++ compiler.
The default target architecture is `sm_86` (RTX 30-series, Ampere). Change it
for other GPUs, for example `sm_75` for Turing or `sm_89` for Ada.

### Windows (CUDA Toolkit + MSVC)

`nvcc` on Windows needs MSVC's `cl.exe`. The Visual Studio Build Tools with
the "Desktop development with C++" workload are enough. `build.ps1` finds
Visual Studio with `vswhere` and loads `vcvars64.bat` for you if `cl.exe`
isn't already set up in your shell.

```powershell
.\build.ps1                      # outputs to .\build\
.\build.ps1 -Arch sm_75          # different GPU
.\build.ps1 -OutDir C:\temp\mm   # different output folder
```

To build by hand from a "x64 Native Tools Command Prompt":

```bat
nvcc -O2 -arch=sm_86 -o matrix_mul_compare.exe matrix_mul_compare.cu
```

### Linux / WSL

```bash
make             # builds build/hello, build/matrix_mul, build/matrix_mul_compare, build/matrix_mul_cpu
make ARCH=sm_75
```

Without a GPU you can still build and run the CPU baseline:

```bash
g++ -O2 -std=c++17 -o matrix_mul_cpu matrix_mul.cpp
```

## Running and expected output

```text
> build\hello.exe
Hello, World from CUDA!

> build\matrix_mul.exe
C[0][0] = 2048.00 (expected: 2048.00)
C[10][10] = 2048.00 (expected: 2048.00)

> build\matrix_mul_cpu.exe
C[0][0] = 2048.00 (expected: 2048.00)
C[10][10] = 2048.00 (expected: 2048.00)
Execution time: <t> seconds

> build\matrix_mul_compare.exe
Matrix size: 1024x1024
CPU time: <t_cpu> seconds
GPU time: <t_gpu> seconds
Speedup: <t_cpu / t_gpu>x
Results match
```

## Results

Test machine: laptop with an **NVIDIA RTX 3080 Ti Laptop GPU (16 GB)**, an
**Intel Core i9-12900HX** and 64 GB DDR5, running Windows 11, CUDA 12.1 and
MSVC 14.29 (VS 2019 Build Tools).

### CPU baseline (`matrix_mul.cpp`, 1024 x 1024, single thread)

These were measured when this repo was put together, while other work was
running on the machine, so treat them as rough.

| Compiler | Flags | Time |
|----------|-------|------|
| g++ 16.2 (MSYS2 UCRT64, Windows) | `-O2` | 0.61 to 0.62 s (3 runs) |
| g++ 11.4 (Ubuntu 22.04 on WSL2) | `-O2` | 2.56 to 3.02 s (2 runs) |
| MSVC 14.29 (`cl`, Windows) | `/O2` | 2.61 s (1 run) |

The same naive loop varies about 4x across compilers and versions. The CPU
side of any "CPU vs GPU" speedup is therefore fragile.

### GPU (run it yourself)

No GPU timings were saved from the original experiments. Fill these in from
your own runs rather than trusting made-up numbers:

| Program | Block size | CPU time | GPU time (incl. alloc + copies + context init) | Speedup | Results |
|---------|-----------|----------|-----------------------------------------------|---------|---------|
| `matrix_mul_compare` | 16 x 16 | _run it_ | _run it_ | _run it_ | _run it_ |

## Ideas for next steps

- Time only the kernel with `cudaEvent_t`, and warm up the context before
  timing.
- Add a shared-memory tiled kernel and compare it with the naive one.
- Compare against `cublasSgemm`.
- Wrap CUDA calls in an error-checking macro.
- Take the matrix size from the command line.

## License

No license has been chosen yet, so all rights are reserved by default.
