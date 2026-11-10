# EXaCTz

EXaCTz is a CUDA-based research implementation for topology-aware lossy
compression of three-dimensional scalar fields. This repository contains
single-GPU and distributed multi-GPU drivers.

> **Research software:** EXaCTz is provided for research and evaluation. It
> is not a stable compression library or a portable file format, and the
> current drivers do not provide a standalone decompression command.

## Included programs

| Program | Source | Description |
| --- | --- | --- |
| `EXaCTz_single` | `src/ExtreM_cuda.cu` | Single-GPU driver |
| `EXaCTz_mpi` | `src/pExtreMz_all_order.cu` | Distributed multi-GPU driver |

## Requirements

- Linux
- NVIDIA GPU and CUDA toolkit
- CMake 3.20 or newer
- C++17-compatible host compiler
- OpenMP
- SZ3 headers
- Zstandard library and `zstd` command-line tool
- ZFP (with CUDA support for CUDA ZFP experiments)
- MGARD-X and cuSZp for the single-GPU driver
- CUDA-aware MPI for the distributed driver

Optional compressor backends may require `mgard-x`, `tthresh`, or `sperr3d`
available on `PATH`. Dependency versions have not been pinned; validate the
toolkit and library versions on the target system before running experiments.

## Build

Choose the CUDA architecture for the target GPU (`80` is suitable for an
A100):

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_PREFIX_PATH="/path/to/sz3;/path/to/zstd;/path/to/zfp;/path/to/mgard;/path/to/cuszp"
cmake --build build -j 8
```

If SZ3 generated headers are installed separately, add
`-DSZ3_EXTRA_INCLUDE_DIR=/path/to/SZ3/build/include`.

Build only one driver when the other toolchain is unavailable:

```bash
# Single GPU only
cmake -S . -B build-single -DEXACTZ_BUILD_MPI=OFF

# MPI only
cmake -S . -B build-mpi -DEXACTZ_BUILD_SINGLE=OFF
```

For MPI builds, set `MPI_CXX_COMPILER` if `mpicxx` is not discoverable.

## Input format

The drivers read a raw native-endian, double-precision scalar field without a
header. The file must contain exactly `nx * ny * nz * 8` bytes. Pass the input
as `file,nx,ny,nz`; filenames must not contain commas.

Set `EXACTZ_WORK_DIR` to a writable directory before running. Run experiments
in separate work directories because some diagnostics use fixed filenames or
append to existing logs.

## Examples

Generate a small test field using only the Python standard library:

```bash
python3 examples/make_toy.py
export EXACTZ_WORK_DIR="$PWD/exactz-work"
```

Run the single-GPU driver:

```text
EXaCTz_single file,nx,ny,nz relative_bound compressor sim q edit_type compression edit_mode [zfp_cuda_cr] [q_study] [experiment_mode]
```

For example:

```bash
./build/EXaCTz_single \
  "$PWD/toy_f64.bin,32,32,32" 1e-4 sz3 0 5 1 1 0
```

Run the distributed driver with one GPU per MPI rank:

```text
EXaCTz_mpi file,nx,ny,nz absolute_bound compressor sim q edit_type compression PX PY PZ preserveFinal result_prefix decp_prefix compression_mode
```

The product `PX*PY*PZ` must equal the MPI rank count. The distributed driver
uses an absolute error bound, while the single-GPU SZ3 path accepts a relative
bound. The `preserveFinal` argument is currently parsed but not used.

```bash
mpiexec -n 2 ./build/EXaCTz_mpi \
  "$PWD/toy_f64.bin,32,32,32" 1e-4 sz3 0 5 1 0 1 1 2 0 smoke "" 1
```

On clusters, use the site's documented GPU-binding options with `srun` or
`mpiexec`. CUDA-aware MPI must be enabled according to the MPI
implementation's documentation.

## Limitations

This repository is a research snapshot. Before using results in a paper or
benchmark, validate the selected CUDA/compiler/dependency versions and verify
the reconstructed field, topology, and actual error bound independently.
Large-grid memory use, concurrent output naming, malformed input handling, and
all compressor configurations have not been exhaustively validated.

## License

EXaCTz is released under the [MIT License](LICENSE). Third-party dependencies
are distributed under their own licenses.
