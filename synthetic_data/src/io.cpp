#include "io.h"
#include <stdlib.h>
#include <string.h>

static FILE *open_path(const char *dir, const char *name)
{
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    return fopen(path, "w");
}

OutputFiles open_output_files(const char *out_dir)
{
    OutputFiles f;
    f.f_xy = open_path(out_dir, "out_Txy.txt");
    f.f_z = open_path(out_dir, "out_Tz.txt");
    f.f_t = open_path(out_dir, "out_Tt.txt");
    f.f_test = open_path(out_dir, "out_test.txt");
    f.f_xz = open_path(out_dir, "out_Txz.txt");
    f.f_yz = open_path(out_dir, "out_Tyz.txt");
    return f;
}

void close_output_files(OutputFiles *files)
{
    fclose(files->f_xy);
    fclose(files->f_z);
    fclose(files->f_t);
    fclose(files->f_test);
    fclose(files->f_xz);
    fclose(files->f_yz);
}

void write_tt_sample(OutputFiles *out, double time,
                     double T0mm, double T1mm, double T3mm)
{
    fprintf(out->f_t, "%e  %.3f  %.3f  %.3f\n", time, T0mm, T1mm, T3mm);
}

void write_peak_snapshots(OutputFiles *out, double T1[Nx1][Ny][Nz],
                          double T2[Nx2][Ny][Nz], int i0, int i00,
                          double dx1, double dx2, const double y[Ny],
                          const double z[Nz])
{
    for (int i = 0; i < Nx1; i++) {
        for (int j = 0; j < Ny; j++) {
            fprintf(out->f_xy, "%e   %e   %.3f\n",
                    (i0 * dx2 - Nx1 / 2 * dx1) + i * dx1, y[j], T1[i][j][0]);
            if (j != 0) {
                fprintf(out->f_xy, "%e   %e   %.3f\n",
                        (i0 * dx2 - Nx1 / 2 * dx1) + i * dx1, -y[j], T1[i][j][0]);
            }
        }
        for (int k = 0; k < Nz; k++) {
            fprintf(out->f_xz, "%e   %e   %.3f\n",
                    (i0 * dx2 - Nx1 / 2 * dx1) + i * dx1, z[k], T1[i][0][k]);
        }
    }
    for (int i = 0; i < Nx2; i++) {
        for (int j = 0; j < Ny; j++) {
            if (i * dx2 < (i0 * dx2 - Nx1 / 2 * dx1) ||
                i * dx2 > (i0 * dx2 + Nx1 / 2 * dx1)) {
                fprintf(out->f_xy, "%e   %e   %.3f\n", i * dx2, y[j], T2[i][j][0]);
                if (j != 0) {
                    fprintf(out->f_xy, "%e   %e   %.3f\n", i * dx2, -y[j], T2[i][j][0]);
                }
            }
        }
        for (int k = 0; k < Nz; k++) {
            if (i * dx2 < (i0 * dx2 - Nx1 / 2 * dx1) ||
                i * dx2 > (i0 * dx2 + Nx1 / 2 * dx1)) {
                fprintf(out->f_xz, "%e   %e   %.3f\n", i * dx2, z[k], T2[i][0][k]);
            }
        }
    }
    for (int k = 0; k < Nz; k++) {
        for (int j = 0; j < Ny; j++) {
            fprintf(out->f_yz, "%e   %e   %.3f\n", y[j], z[k], T1[Nx1 / 2][j][k]);
            if (j != 0) {
                fprintf(out->f_yz, "%e   %e   %.3f\n", -y[j], z[k], T1[Nx1 / 2][j][k]);
            }
        }
    }
}
