#include "grid.h"
#include "config.h"
#include <math.h>

void generate_grid(double y[Ny], double z[Nz])
{
    y[0] = 0;
    for (int j = 0; j < Ny - 1; j++) {
        double dy = 30e-6 + pow(j + 1, 2.0) * 5e-7;
        if (dy > 300e-6) dy = 300e-6;
        y[j + 1] = y[j] + dy;
    }

    z[0] = 0;
    for (int k = 0; k < Nz - 1; k++) {
        double dz = 30e-6 + pow(k + 1, 1.8) * 2e-7;
        if (dz > 100e-6) dz = 100e-6;
        z[k + 1] = z[k] + dz;
    }
}

void find_depth_indices(const double z[Nz], int *k_0mm, int *k_1mm, int *k_3mm)
{
    *k_0mm = 0;
    *k_1mm = 0;
    *k_3mm = 0;
    for (int k = 0; k < Nz; k++) {
        if (z[k] >= 1e-3 && *k_1mm == 0) *k_1mm = k;
        if (z[k] >= 3e-3 && *k_3mm == 0) *k_3mm = k;
    }
}
