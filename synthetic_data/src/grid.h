#ifndef GRID_H
#define GRID_H

/* Non-uniform y and z node coordinates */
void generate_grid(double y[Ny], double z[Nz]);

/* Indices for sensor depths 0 mm, 1 mm, 3 mm */
void find_depth_indices(const double z[Nz], int *k_0mm, int *k_1mm, int *k_3mm);

#endif
