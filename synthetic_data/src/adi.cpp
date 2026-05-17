#include "adi.h"
#include "material.h"
#include <math.h>

double laser_flux(double time1, double t_peak, double pulse_sigma, double Q0,
                  int i, int j, double dx1, double delta, double r0,
                  const double y[Ny])
{
    return Q0 * exp(-pow(time1 - t_peak, 2) / (2.0 * pow(pulse_sigma, 2))) *
           exp(-(pow((i - Nx1 / 2) * dx1 - delta, 2) + pow(y[j], 2)) * LN2 / pow(r0, 2));
}

double volumetric_source(double q0_ij, double z_k, int use_volumetric)
{
    if (!use_volumetric) {
        return 0.0;
    }
    return q0_ij * ALPHA_ABS * exp(-ALPHA_ABS * z_k);
}

void adi_z_local(double T1[Nx1][Ny][Nz], double q0[Nx1][Ny],
                 const double z[Nz], double dt, int nymax, int nzmax,
                 int i, int j, int use_volumetric)
{
    static double ATz[Nz], BTz[Nz];
    double JT, ks0, ks1, cs, a, b, c, d, g;

    if (use_volumetric == 0) {
        JT = q0[i][j] / (z[1] - z[0]);
    } else {
        JT = volumetric_source(q0[i][j], z[0], use_volumetric);
    }
    ks1 = hcond(T1[i][j][0]);
    cs = hcap(T1[i][j][0]);
    b = -(ks1 / pow(z[1] - z[0], 2) + cs / dt);
    c = ks1 / pow(z[1] - z[0], 2);
    d = -(T1[i][j][0] * cs / dt + JT);
    ATz[0] = -c / b;
    BTz[0] = d / b;

    for (int k = 1; k < nzmax; k++) {
        if (use_volumetric == 0) {
            JT = 0;
        } else {
            JT = volumetric_source(q0[i][j], z[k], use_volumetric);
        }
        ks0 = hcond(T1[i][j][k]);
        ks1 = hcond(T1[i][j][k + 1]);
        cs = hcap(T1[i][j][k]);
        a = ks0 / pow(z[k] - z[k - 1], 2);
        b = -(cs / dt + ks0 / pow(z[k] - z[k - 1], 2) +
              ks1 / ((z[k + 1] - z[k]) * (z[k] - z[k - 1])));
        c = ks1 / ((z[k + 1] - z[k]) * (z[k] - z[k - 1]));
        d = -(T1[i][j][k] * cs / dt + JT);
        g = a * ATz[k - 1] + b;
        ATz[k] = -c / g;
        BTz[k] = (d - a * BTz[k - 1]) / g;
    }
    ks1 = hcond(T1[i][j][nzmax - 1]);
    cs = hcap(T1[i][j][nzmax - 1]);
    a = ks1 / pow(z[nzmax - 1] - z[nzmax - 2], 2);
    b = -ks1 / pow(z[nzmax - 1] - z[nzmax - 2], 2) - cs / dt;
    d = -cs / dt * T1[i][j][nzmax - 1];
    T1[i][j][nzmax - 1] = (d - a * BTz[nzmax - 2]) / (b + a * ATz[nzmax - 2]);
    for (int k = nzmax - 2; k >= 0; k--) {
        T1[i][j][k] = ATz[k] * T1[i][j][k + 1] + BTz[k];
    }
}

void adi_y_local(double T1[Nx1][Ny][Nz], const double y[Ny],
                 double dt, int nx, int nymax, int nzmax, int i, int k)
{
    static double ATy[Ny], BTy[Ny];
    double JT = 0, ks0, ks1, cs, a, b, c, d, g;

    ks0 = hcond(T1[i][0][k]);
    cs = hcap(T1[i][0][k]);
    b = -(ks0 / pow(y[1] - y[0], 2) + cs / dt);
    c = ks0 / pow(y[1] - y[0], 2);
    d = -(T1[i][0][k] * cs / dt + JT);
    ATy[0] = -c / b;
    BTy[0] = d / b;

    for (int j = 1; j < nymax; j++) {
        ks0 = hcond(T1[i][j][k]);
        ks1 = hcond(T1[i][j + 1][k]);
        cs = hcap(T1[i][j][k]);
        a = ks0 / pow(y[j] - y[j - 1], 2);
        b = -(cs / dt + ks0 / pow(y[j] - y[j - 1], 2) +
              ks1 / ((y[j + 1] - y[j]) * (y[j] - y[j - 1])));
        c = ks1 / ((y[j + 1] - y[j]) * (y[j] - y[j - 1]));
        d = -(T1[i][j][k] * cs / dt + JT);
        g = a * ATy[j - 1] + b;
        ATy[j] = -c / g;
        BTy[j] = (d - a * BTy[j - 1]) / g;
    }
    cs = hcap(T1[i][nymax][k]);
    ks0 = hcond(T1[i][nymax][k]);
    a = ks0 / pow(y[nymax] - y[nymax - 1], 2);
    b = -(cs / dt + ks0 / pow(y[nymax] - y[nymax - 1], 2));
    d = -(T1[i][nymax][k] * cs / dt + JT);
    T1[i][nymax][k] = (d - a * BTy[nymax - 1]) / (b + a * ATy[nymax - 1]);
    for (int j = nymax - 1; j >= 0; j--) {
        T1[i][j][k] = ATy[j] * T1[i][j + 1][k] + BTy[j];
    }
}

void adi_x_local(double T1[Nx1][Ny][Nz], double dx1, double dt,
                 int nx, int nymax, int nzmax, int j, int k)
{
    static double ATx[Nx1], BTx[Nx1];
    double JT = 0, ks0, ks1, cs, a, b, c, d, g;

    ks0 = hcond(T1[0][j][k]);
    cs = hcap(T1[0][j][k]);
    b = -(ks0 / pow(dx1, 2) + cs / dt);
    c = ks0 / pow(dx1, 2);
    d = -(T1[0][j][k] * cs / dt + JT);
    ATx[0] = -c / b;
    BTx[0] = d / b;

    for (int i = 1; i < nx - 1; i++) {
        ks0 = hcond(T1[i][j][k]);
        ks1 = hcond(T1[i + 1][j][k]);
        cs = hcap(T1[i][j][k]);
        a = ks0 / pow(dx1, 2);
        b = -(cs / dt + (ks0 + ks1) / pow(dx1, 2));
        c = ks1 / pow(dx1, 2);
        d = -(T1[i][j][k] * cs / dt + JT);
        g = a * ATx[i - 1] + b;
        ATx[i] = -c / g;
        BTx[i] = (d - a * BTx[i - 1]) / g;
    }
    cs = hcap(T1[nx - 1][j][k]);
    ks0 = hcond(T1[nx - 1][j][k]);
    a = ks0 / pow(dx1, 2);
    b = -(cs / dt + ks0 / pow(dx1, 2));
    d = -(T1[nx - 1][j][k] * cs / dt + JT);
    T1[nx - 1][j][k] = (d - a * BTx[nx - 2]) / (b + a * ATx[nx - 2]);
    for (int i = nx - 2; i >= 0; i--) {
        T1[i][j][k] = ATx[i] * T1[i + 1][j][k] + BTx[i];
    }
}

void adi_z_global(double T2[Nx2][Ny][Nz], const double z[Nz],
                  double dt, int i, int j, int nzmax)
{
    static double ATz[Nz], BTz[Nz];
    double JT = 0, ks0, ks1, cs, a, b, c, d, g;

    ATz[0] = 1;
    BTz[0] = 0;
    for (int k = 1; k < nzmax - 1; k++) {
        ks0 = hcond(T2[i][j][k]);
        ks1 = hcond(T2[i][j][k + 1]);
        cs = hcap(T2[i][j][k]);
        a = ks0 / pow(z[k] - z[k - 1], 2);
        b = -(cs / dt + ks0 / pow(z[k] - z[k - 1], 2) +
              ks1 / ((z[k + 1] - z[k]) * (z[k] - z[k - 1])));
        c = ks1 / ((z[k + 1] - z[k]) * (z[k] - z[k - 1]));
        d = -(T2[i][j][k] * cs / dt + JT);
        g = a * ATz[k - 1] + b;
        ATz[k] = -c / g;
        BTz[k] = (d - a * BTz[k - 1]) / g;
    }
    ks1 = hcond(T2[i][j][nzmax - 1]);
    cs = hcap(T2[i][j][nzmax - 1]);
    a = ks1 / pow(z[nzmax - 1] - z[nzmax - 2], 2);
    b = -ks1 / pow(z[nzmax - 1] - z[nzmax - 2], 2) - cs / dt;
    d = -cs / dt * T2[i][j][nzmax - 1];
    T2[i][j][nzmax - 1] = (d - a * BTz[nzmax - 2]) / (b + a * ATz[nzmax - 2]);
    for (int k = nzmax - 2; k >= 0; k--) {
        T2[i][j][k] = ATz[k] * T2[i][j][k + 1] + BTz[k];
    }
}

void adi_y_global(double T2[Nx2][Ny][Nz], const double y[Ny],
                  double dt, int i, int k, int nymax, int nzmax)
{
    static double ATy[Ny], BTy[Ny];
    double JT = 0, ks0, ks1, cs, a, b, c, d, g;

    (void)nzmax;
    ATy[0] = 1;
    BTy[0] = 0;
    for (int j = 1; j < nymax - 1; j++) {
        ks0 = hcond(T2[i][j][k]);
        ks1 = hcond(T2[i][j + 1][k]);
        cs = hcap(T2[i][j][k]);
        a = ks0 / pow(y[j] - y[j - 1], 2);
        b = -(cs / dt + ks0 / pow(y[j] - y[j - 1], 2) +
              ks1 / ((y[j + 1] - y[j]) * (y[j] - y[j - 1])));
        c = ks1 / ((y[j + 1] - y[j]) * (y[j] - y[j - 1]));
        d = -(T2[i][j][k] * cs / dt + JT);
        g = a * ATy[j - 1] + b;
        ATy[j] = -c / g;
        BTy[j] = (d - a * BTy[j - 1]) / g;
    }
    ks1 = hcond(T2[i][nymax - 1][k]);
    cs = hcap(T2[i][nymax - 1][k]);
    a = ks1 / pow(y[nymax - 1] - y[nymax - 2], 2);
    b = -ks1 / pow(y[nymax - 1] - y[nymax - 2], 2) - cs / dt;
    d = -cs / dt * T2[i][nymax - 1][k];
    T2[i][nymax - 1][k] = (d - a * BTy[nymax - 2]) / (b + a * ATy[nymax - 2]);
    for (int j = nymax - 2; j >= 0; j--) {
        T2[i][j][k] = ATy[j] * T2[i][j + 1][k] + BTy[j];
    }
}

void adi_x_global(double T2[Nx2][Ny][Nz], double dx2, double dt,
                  int j, int k, int nxmax, int nymax, int nzmax)
{
    static double ATx[Nx2], BTx[Nx2];
    double JT = 0, ks0, ks1, cs, a, b, c, d, g;

    (void)nymax;
    (void)nzmax;
    ATx[0] = 1;
    BTx[0] = 0;
    for (int i = 1; i < nxmax - 1; i++) {
        ks0 = hcond(T2[i][j][k]);
        ks1 = hcond(T2[i + 1][j][k]);
        cs = hcap(T2[i][j][k]);
        a = ks0 / pow(dx2, 2);
        b = -(cs / dt + (ks0 + ks1) / pow(dx2, 2));
        c = ks1 / pow(dx2, 2);
        d = -(T2[i][j][k] * cs / dt + JT);
        g = a * ATx[i - 1] + b;
        ATx[i] = -c / g;
        BTx[i] = (d - a * BTx[i - 1]) / g;
    }
    ks1 = hcond(T2[nxmax - 1][j][k]);
    cs = hcap(T2[nxmax - 1][j][k]);
    a = ks1 / pow(dx2, 2);
    b = -ks1 / pow(dx2, 2) - cs / dt;
    d = -cs / dt * T2[nxmax - 1][j][k];
    T2[nxmax - 1][j][k] = (d - a * BTx[nxmax - 2]) / (b + a * ATx[nxmax - 2]);
    for (int i = nxmax - 2; i >= 0; i--) {
        T2[i][j][k] = ATx[i] * T2[i + 1][j][k] + BTx[i];
    }
}
