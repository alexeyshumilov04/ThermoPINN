#ifndef ADI_H
#define ADI_H

#include "config.h"

/* Gaussian heat flux at surface (W/m²) for local grid node (i,j) */
double laser_flux(double time1, double t_peak, double pulse_sigma, double Q0,
                  int i, int j, double dx1, double delta, double r0,
                  const double y[Ny]);

/* Volumetric heat source JT [W/m³] from Beer–Lambert */
double volumetric_source(double q0_ij, double z_k, int use_volumetric);

/* One ADI sub-step: implicit sweep along z on local grid T1 */
void adi_z_local(double T1[Nx1][Ny][Nz], double q0[Nx1][Ny],
                 const double z[Nz], double dt, int nymax, int nzmax,
                 int i, int j, int use_volumetric);

/* Implicit sweeps along y and x on local grid (when domain is small enough) */
void adi_y_local(double T1[Nx1][Ny][Nz], const double y[Ny],
                 double dt, int nx, int nymax, int nzmax, int i, int k);

void adi_x_local(double T1[Nx1][Ny][Nz], double dx1, double dt,
                 int nx, int nymax, int nzmax, int j, int k);

/* Cooling: z / y / x sweeps on global grid T2 (no heat source) */
void adi_z_global(double T2[Nx2][Ny][Nz], const double z[Nz],
                  double dt, int i, int j, int nzmax);

void adi_y_global(double T2[Nx2][Ny][Nz], const double y[Ny],
                  double dt, int i, int k, int nymax, int nzmax);

void adi_x_global(double T2[Nx2][Ny][Nz], double dx2, double dt,
                  int j, int k, int nxmax, int nymax, int nzmax);

#endif
