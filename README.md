# EXaCTz source release candidate 0.1.0-rc1

This candidate packages the authors' single-GPU and distributed multi-GPU research drivers. It is prepared for review and Linux/NVIDIA validation; it is not yet a tested public release.

| Program | Source |
|---|---|
| `EXaCTz_single` | `src/ExtreM_cuda.cu` |
| `EXaCTz_mpi` | `src/pExtreMz_all_order.cu` |

Only these drivers and their local header are included. Experimental variants, old binaries, datasets, and benchmark logs are excluded. `SOURCE_MANIFEST.json` records the original working-tree hashes and the packaged hashes. The original workspace is unchanged.

## Requirements and build

Use Linux, an NVIDIA GPU, a compatible CUDA toolkit/host C++ compiler, CMake >= 3.20, SZ3 headers, Zstandard (library and `zstd` CLI), ZFP, and OpenMP. MPI builds additionally require CUDA-aware MPI: the source passes GPU buffers to MPI. The single-GPU source unconditionally includes MGARD-X and cuSZp; both development installations are required even when selecting SZ3. ZFP CUDA mode requires ZFP built with CUDA support. Optional external backends need `mgard-x`, `tthresh`, or `sperr3d` on PATH. Dependency versions have not yet been pinned by a validated build.

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_PREFIX_PATH="/path/to/sz3;/path/to/zstd;/path/to/zfp;/path/to/mgard;/path/to/cuszp"
cmake --build build -j 8
```

Set the CUDA architecture for your hardware (`80` is for A100). If SZ3 generated headers are in a separate directory, add `-DSZ3_EXTRA_INCLUDE_DIR=/path/to/SZ3/build/include`. For MPI only, add `-DEXACTZ_BUILD_SINGLE=OFF`. For single GPU only, add `-DEXACTZ_BUILD_MPI=OFF`. If the MPI wrapper is not discoverable, specify `-DMPI_CXX_COMPILER=/path/to/mpicxx`. Set runtime library search paths according to your dependency installation. This CMake build has not yet been compiled with CUDA.

## Input and command-line interface

Input is a raw native-endian double-precision scalar field, without a header. The expected size is `nx * ny * nz * 8` bytes. Supply dimensions as `file,nx,ny,nz`; filenames must not contain commas. Inspect each driver's experiment flags before changing them. The examples below use the existing scripts' `sim=0`, `q=5`, `edit_type=1` settings; they are smoke-test starting points, not a validated paper reproduction configuration.

Set `EXACTZ_WORK_DIR` to a writable absolute path, shared by all MPI ranks. Output subdirectories are created there. For single-GPU diagnostics, the existing `SCRATCH/ExtreM_z_outputs` override takes precedence if `SCRATCH` is set. Run each experiment in its own directory because some logs use fixed names or append to existing files.

Single GPU:

```text
EXaCTz_single file,nx,ny,nz relative_bound compressor sim q edit_type compression edit_mode [zfp_cuda_cr] [q_study] [experiment_mode]
```

For SZ3, the single-GPU driver uses a relative error bound; the driver derives an absolute bound from the data range. `q` must be positive. `compression=1` runs compression; `edit_mode` selects an existing repair mode. Optional defaults are `10`, `0`, `0`.

```bash
export EXACTZ_WORK_DIR="$PWD/exactz-work"
python3 examples/make_toy.py
./build/EXaCTz_single "$PWD/toy_f64.bin,32,32,32" 1e-4 sz3 0 5 1 1 0
```

Multi-GPU:

```text
EXaCTz_mpi file,nx,ny,nz absolute_bound compressor sim q edit_type compression PX PY PZ preserveFinal result_prefix decp_prefix compression_mode
```

All 14 positional arguments are required. `PX*PY*PZ` must equal the MPI rank count. Allocate one GPU per rank. Use `compression_mode=1` for local core-only compression. Mode 0 uses global/precomputed data paths and requires additional validation. The multi-GPU bound is absolute, so do not compare it directly to the single-GPU relative-bound argument. `preserveFinal` is parsed by this snapshot but is not subsequently used; it should not be interpreted as enabling a guarantee. The MPI driver has no supported standalone decompressor command.

```bash
export EXACTZ_WORK_DIR="$PWD/exactz-work"
mpiexec -n 2 ./build/EXaCTz_mpi "$PWD/toy_f64.bin,32,32,32" 1e-4 sz3 0 5 1 0 1 1 2 0 smoke "" 1
```

On a Slurm GPU allocation, replace `mpiexec -n 2` with your site's GPU-binding `srun` command. For Perlmutter CUDA-aware MPICH, use `MPICH_GPU_SUPPORT_ENABLED=1` and the site's documented GPU allocation/binding settings.

## Status and limitations

See `RELEASE_REVIEW.md` for the checks needed before public release. This is a research driver snapshot, not a stable compression/decompression library or portable file format. Do not interpret a successful process exit as proof of topology preservation. Independent topology comparison, reconstruction, and error-bound verification are required on the chosen configurations.

No project license was found in the workspace. The copyright owners must choose and approve a license before public distribution; no license has been invented for this candidate. Dependencies retain their own licenses.
