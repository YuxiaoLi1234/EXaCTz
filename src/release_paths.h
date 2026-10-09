#pragma once
#include <cstdlib>
#include <string>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
// Work directory must be shared across MPI ranks.
inline std::string exactz_scratch() {
    const char* value = std::getenv("EXACTZ_WORK_DIR");
    return value && *value ? std::string(value) : std::string("./exactz-work");
}
inline void exactz_prepare_work() {
    const auto base = std::filesystem::path(exactz_scratch());
    for (const auto* sub : {"tmp", "MSCz", "datasets", "datasets/upsamples", "ExtreM_z_outputs"})
        std::filesystem::create_directories(base / sub);
}
