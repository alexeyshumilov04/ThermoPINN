#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#include "config.h"
#include "material.h"
#include "grid.h"
#include "adi.h"
#include "io.h"

int main(void)
{
    static double T1[Nx1][Ny][Nz], T2[Nx2][Ny][Nz];
    static double q0[Nx1][Ny];
    static double z[Nz], y[Ny];

    double v, pulse_period, Ep, pulse_fwhm, pulse_sigma, t_peak, r0, A, Q0, f;
    double T0 = T0_DEFAULT;
    int use_volumetric = USE_VOLUMETRIC_DEFAULT;
    int Np = N_PULSES_DEFAULT;
    int np0 = 1;

    double dx1, dx2, dt1, dt2, time = 0, time1, time2;
    double delta = 0, Tmax = 0;
    int i0, ind, i00 = Nx2 / 2;
    int ni, k_0mm, k_1mm, k_3mm;

    int Nzmax1 = 80, Nymax1 = 80;
    int Nxmax2 = 80;
    int Nymax2 = 80, Nzmax2 = 80;

    printf("=======================================================\n");
    printf("  3 GAUSSIAN PULSES (FWHM = 3 s) — agar gel phantom\n");
    printf("=======================================================\n\n");
    printf("Input v (mm/s) [0 for stationary]: ");
    scanf("%lf", &v);
    printf("Input period between pulses (s): ");
    scanf("%lf", &pulse_period);
    printf("Input energy per pulse (J): ");
    scanf("%lf", &Ep);

    pulse_fwhm = PULSE_FWHM_DEFAULT;
    pulse_sigma = pulse_fwhm / 2.355;
    t_peak = 3.0 * pulse_sigma;
    f = 1.0 / pulse_period;
    r0 = R0_DEFAULT;
    A = A_DEFAULT;

    dx1 = r0 / 10.0;
    dx2 = r0 / 5.0;
    Nxmax2 = i00 + 3 * (int)floor(r0 / dx2);

    dt1 = DT_UNIFORM;
    dt2 = DT_UNIFORM;
    Q0 = Ep / (PI * pow(r0, 2) * sqrt(2.0 * PI) * pulse_sigma);

    printf("\n=== Configuration ===\n");
    printf("Pulses: %d\n", Np);
    printf("Pulse FWHM: %.1f s, sigma: %.3f s\n", pulse_fwhm, pulse_sigma);
    printf("Period: %.1f s, Energy/pulse: %.3e J\n", pulse_period, Ep);
    printf("Velocity: %.2f mm/s\n", v);
    printf("Beam FWHM diameter: %.1f mm\n", r0 * 2e3);
    printf("Wavelength: %.0f nm, alpha_abs: %.2f m^-1\n", LAMBDA * 1e9, ALPHA_ABS);
    printf("Volumetric absorption: %s\n",
           use_volumetric ? "ENABLED (Beer-Lambert)" : "DISABLED");
    printf("=====================\n\n");

    clock_t timeprog = clock();
    v = v / 1e3;
    i0 = i00;

    generate_grid(y, z);
    find_depth_indices(z, &k_0mm, &k_1mm, &k_3mm);

    printf("Grid: Y 0..%.2f mm (%d pts), Z 0..%.2f mm (%d pts)\n",
           y[Ny - 1] * 1e3, Ny, z[Nz - 1] * 1e3, Nz);

    for (int k = 0; k < Nz; k++) {
        for (int j = 0; j < Ny; j++) {
            for (int i = 0; i < Nx1; i++) T1[i][j][k] = T0;
            for (int i = 0; i < Nx2; i++) T2[i][j][k] = T0;
        }
    }

    const char *out_dir = "../data";
    OutputFiles out = open_output_files(out_dir);

    for (int np = 0; np < Np; np++) {
        time1 = 0;
        time2 = 0;
        ind = 1;

        /* Heating phase: Gaussian pulse */
        while (time1 < 6.0 * pulse_sigma) {
#pragma omp parallel for schedule(dynamic) num_threads(Ntr)
            for (int i = 0; i < Nx1; i++) {
                for (int j = 0; j <= Nymax1; j++) {
                    q0[i][j] = A * laser_flux(time1, t_peak, pulse_sigma, Q0,
                                              i, j, dx1, delta, r0, y);
                    adi_z_local(T1, q0, z, dt1, Nymax1, Nzmax1, i, j, use_volumetric);
                }
            }

            if (r0 <= 10 * sqrt(2.5 * pulse_sigma * hcond(T0) / hcap(T0))) {
                for (int i = 0; i < Nx1; i++) {
                    for (int k = 0; k < Nzmax1; k++) {
                        adi_y_local(T1, y, dt1, Nx1, Nymax1, Nzmax1, i, k);
                    }
                }
                for (int k = 0; k < Nzmax1; k++) {
                    for (int j = 0; j <= Nymax1; j++) {
                        adi_x_local(T1, dx1, dt1, Nx1, Nymax1, Nzmax1, j, k);
                    }
                }
            }

            time1 += dt1;
            time += dt1;
            write_tt_sample(&out, time,
                            T1[Nx1 / 2][0][k_0mm], T1[Nx1 / 2][0][k_1mm],
                            T1[Nx1 / 2][0][k_3mm]);

            if (np == np0 && time1 > pulse_sigma && ind == 1) {
                if (T1[Nx1 / 2][0][0] < Tmax) {
                    write_peak_snapshots(&out, T1, T2, i0, i00, dx1, dx2, y, z);
                    ind = 0;
                }
            }
            Tmax = T1[Nx1 / 2][0][0];
        }

        /* Copy local → global grid */
        for (int j = 0; j < Ny; j++) {
            for (int k = 0; k < Nz; k++) {
                for (int i = i0; i < Nx2; i++) {
                    if (3 * (i - i0) < Nx1 / 2) {
                        T2[i][j][k] = T1[Nx1 / 2 + 3 * (i - i0)][j][k];
                        T2[2 * i0 - 1 - i][j][k] = T1[Nx1 / 2 - 3 * (i + 1 - i0)][j][k];
                    }
                }
            }
        }

        /* Cooling between pulses */
        while (time2 < (1.0 / f - time1)) {
#pragma omp parallel for schedule(dynamic) num_threads(Ntr)
            for (int i = 0; i < Nxmax2; i++) {
                for (int j = 0; j < Nymax2; j++) {
                    adi_z_global(T2, z, dt2, i, j, Nzmax2);
                }
            }
#pragma omp parallel for schedule(dynamic) num_threads(Ntr)
            for (int i = 0; i < Nxmax2; i++) {
                for (int k = 0; k < Nzmax2; k++) {
                    adi_y_global(T2, y, dt2, i, k, Nymax2, Nzmax2);
                }
            }
#pragma omp parallel for schedule(dynamic) num_threads(Ntr)
            for (int j = 0; j < Nymax2; j++) {
                for (int k = 0; k < Nzmax2; k++) {
                    adi_x_global(T2, dx2, dt2, j, k, Nxmax2, Nymax2, Nzmax2);
                }
            }

            time2 += dt2;
            time += dt2;
            write_tt_sample(&out, time,
                            T2[i0][0][k_0mm], T2[i0][0][k_1mm], T2[i0][0][k_3mm]);

            if (T2[Nxmax2 - 1][0][0] > T0 + 1) Nxmax2++;
            if (T2[i00][Nymax2 - 1][0] > T0 + 1) Nymax2++;
            if (T2[i00][0][Nzmax2 - 1] > T0 + 1) Nzmax2++;
            if (Nxmax2 > Nx2) Nxmax2 = Nx2;
            if (Nymax2 > Ny) Nymax2 = Ny;
            if (Nzmax2 > Nz) Nzmax2 = Nz;
        }

        i0 = i00 + (np + 1) * v / f / dx2;
        delta = i00 * dx2 + (np + 1) * v / f - i0 * dx2;
        if ((i0 * dx2 + 3 * r0) > Nxmax2 * dx2) {
            Nxmax2 = i0 + 3 * (int)floor(r0 / dx2);
        }
        if (Nxmax2 > Nx2) Nxmax2 = Nx2;

        /* Global → local interpolation */
        for (int i = Nx1 / 2; i < Nx1; i++) {
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    ni = i0 + (i - Nx1 / 2) / 3;
                    T1[i][j][k] = T2[ni][j][k] +
                        (dx1 * (i - Nx1 / 2) - (ni - i0) * dx2) *
                        (T2[ni + 1][j][k] - T2[ni][j][k]) / dx2;
                    ni = i0 - (i - Nx1 / 2) / 3 - 1;
                    T1[Nx1 - i - 1][j][k] = T2[ni][j][k] +
                        ((i0 - ni) * dx2 - (i - Nx1 / 2 + 1) * dx1) *
                        (T2[ni + 1][j][k] - T2[ni][j][k]) / dx2;
                }
            }
        }
        printf("executing...  %.2f%%\r", 100.0 * (np + 1) / Np);
        fflush(stdout);
    }

    /* Extended cooling after last pulse */
    double target_time = 2.0 * pulse_period + 6.0 * pulse_sigma + 10.0;
    printf("\nExtended cooling to %.1f s...\n", target_time);
    while (time < target_time) {
#pragma omp parallel for schedule(dynamic) num_threads(Ntr)
        for (int i = 0; i < Nxmax2; i++) {
            for (int j = 0; j < Nymax2; j++) {
                adi_z_global(T2, z, dt2, i, j, Nzmax2);
            }
        }
#pragma omp parallel for schedule(dynamic) num_threads(Ntr)
        for (int i = 0; i < Nxmax2; i++) {
            for (int k = 0; k < Nzmax2; k++) {
                adi_y_global(T2, y, dt2, i, k, Nymax2, Nzmax2);
            }
        }
#pragma omp parallel for schedule(dynamic) num_threads(Ntr)
        for (int j = 0; j < Nymax2; j++) {
            for (int k = 0; k < Nzmax2; k++) {
                adi_x_global(T2, dx2, dt2, j, k, Nxmax2, Nymax2, Nzmax2);
            }
        }
        time += dt2;
        write_tt_sample(&out, time,
                        T2[i0][0][k_0mm], T2[i0][0][k_1mm], T2[i0][0][k_3mm]);
        if (T2[Nxmax2 - 1][0][0] > T0 + 1) Nxmax2++;
    }

    close_output_files(&out);
    timeprog = clock() - timeprog;
    printf("\nDone. Final time: %.1f s, wall time: %.2f s\n",
           time, (double)timeprog / CLOCKS_PER_SEC);
    return 0;
}
