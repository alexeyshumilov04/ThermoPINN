#ifndef CONFIG_H
#define CONFIG_H

/* Grid dimensions (agar gel phantom) */
#define Nx1 102
#define Ny 150
#define Nz 180
#define Nx2 200

/* OpenMP threads */
#define Ntr 1

/* Physical constants */
#define PI 3.14159
#define LN2 0.693147

/* Laser: 980 nm in water/agar */
#define LAMBDA 980e-9
#define ALPHA_ABS 48.2

/* Default simulation parameters */
#define T0_DEFAULT 291.0
#define PULSE_FWHM_DEFAULT 3.0
#define R0_DEFAULT 1.0e-3
#define A_DEFAULT 0.4
#define USE_VOLUMETRIC_DEFAULT 1
#define N_PULSES_DEFAULT 3
#define DT_UNIFORM 0.0005

#endif
