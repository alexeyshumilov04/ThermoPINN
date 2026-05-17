#ifndef IO_H
#define IO_H

#include "config.h"
#include <stdio.h>

typedef struct {
    FILE *f_xy;
    FILE *f_z;
    FILE *f_t;
    FILE *f_test;
    FILE *f_xz;
    FILE *f_yz;
} OutputFiles;

OutputFiles open_output_files(const char *out_dir);
void close_output_files(OutputFiles *files);

/* Time series at beam centre for three depths */
void write_tt_sample(OutputFiles *out, double time,
                     double T0mm, double T1mm, double T3mm);

/* Spatial snapshots at peak temperature (2nd pulse) */
void write_peak_snapshots(OutputFiles *out, double T1[Nx1][Ny][Nz],
                          double T2[Nx2][Ny][Nz], int i0, int i00,
                          double dx1, double dx2, const double y[Ny],
                          const double z[Nz]);

#endif
