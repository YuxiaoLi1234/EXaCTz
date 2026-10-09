# Release review

Prepared 2026-10-09 from the current working tree, not only the last Git commit.

## Changes in this candidate

- Package only the two author-designated CUDA drivers and UnionFind.h.
- Add a separate CMake build with single-GPU/MPI switches.
- Replace active personal scratch paths with EXACTZ_WORK_DIR (default ./exactz-work); create required directories.
- Replace personal optional executable fallbacks with PATH names.
- Add explicit standard-library includes rather than relying on dependency headers.
- Add the missing single-GPU argument-count guard; correct MPI guard and usage to require all 14 positional arguments.
- Keep numerical kernels and repair logic unchanged. Portability changes still require target-system validation.

## Verified locally

- Both requested source files and their local header exist.
- Original CMake builds src/ExtreM.cpp, not the requested CUDA drivers.
- No nvcc is available on this Mac; CUDA compilation and execution are unavailable.
- Source manifests identify the uncommitted originals and packaged copies.
- Toy-data generator produces the expected byte count using only Python's standard library.

## Required before public release

- [ ] Choose/approve the project license and review source ownership.
- [ ] Build both programs on Linux/NVIDIA; record CUDA, host compiler, MPI, and compressor versions and complete build logs.
- [ ] Run single GPU and MPI with 1 and 2 ranks, plus a representative paper dataset/configuration.
- [ ] Compare repaired field topology independently against the original and verify maximum absolute error against the actual absolute bound; save diagnostic logs and output provenance.
- [ ] Verify reconstruction and stored correction data; determine a supported public decompression workflow. Current drivers do not expose a stable encode/decode CLI.
- [ ] Audit large-grid integer products/casts and memory limits before claiming support for large distributed inputs.
- [ ] Check unknown compressor IDs, invalid numeric arguments, malformed dimensions, missing/truncated input, and dependency failures. Argument-count fixes do not provide full input validation.
- [ ] Validate both compression modes and final output semantics. MPI preserveFinal is currently unused; do not advertise it as a supported option.
- [ ] Audit shared temporary files and output naming for concurrent jobs; use separate work directories meanwhile.
- [ ] Review shell-based optional backends and zstd CLI paths; paths containing shell metacharacters are not validated by these research drivers.
- [ ] Confirm the publication citation and paper-matching source version with the authors.
- [ ] Select the destination repository, version/tag, and release text; publish after review.

No target-system build or runtime results are claimed by this candidate.
