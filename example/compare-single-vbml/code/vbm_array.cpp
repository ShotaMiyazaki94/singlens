// Array marshaling only: all numerical methods resolve to the installed wheel.
// This header must belong to the same verified distribution as that binary.
// Called via PyDLL with the GIL held: ESPLMag2 has shared static scratch state.
#include "VBMicrolensingLibrary.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>

extern "C" {

void* vbm_array_create(const char* table, double tolerance, int minimum_annuli) noexcept {
    try {
        auto engine = std::make_unique<VBMicrolensing>();
        engine->Tol = engine->RelTol = tolerance;
        engine->minannuli = minimum_annuli;
        engine->astrometry = false;
        engine->LoadESPLTable(table);
        return engine.release();
    } catch (...) {
        return nullptr;
    }
}

void vbm_array_destroy(void* handle) noexcept {
    delete static_cast<VBMicrolensing*>(handle);
}

int vbm_array_magnifications(void* handle, const double* rho, const double* a1,
                             const double* u, std::size_t walkers, std::size_t epochs,
                             double* out) noexcept {
    try {
        auto& engine = *static_cast<VBMicrolensing*>(handle);
        const double nan = std::numeric_limits<double>::quiet_NaN();
        for (std::size_t walker = 0; walker < walkers; ++walker) {
            const std::size_t offset = walker * epochs;
            bool valid = std::isfinite(rho[walker]) && rho[walker] >= 1e-150 &&
                rho[walker] <= 100 && std::isfinite(a1[walker]) &&
                a1[walker] >= 0 && a1[walker] <= 1;
            engine.a1 = a1[walker];
            for (std::size_t epoch = 0; valid && epoch < epochs; ++epoch) {
                const double separation = u[offset + epoch];
                valid = std::isfinite(separation) && separation >= 0 && separation <= 1e150;
                if (valid) {
                    out[offset + epoch] = engine.ESPLMag2(separation, rho[walker]);
                }
            }
            if (!valid) {
                std::fill(out + offset, out + offset + epochs, nan);
            }
        }
        return 0;
    } catch (...) {
        return -1;
    }
}

}
