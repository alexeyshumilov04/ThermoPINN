#include "material.h"

/* Agar gel: ~0.6 W/(m·K), approximately constant */
double hcond(double T)
{
    (void)T;
    return 0.6;
}

/* Agar: rho ~ 1000 kg/m³, Cp ~ 4000 J/(kg·K) */
double hcap(double T)
{
    (void)T;
    const double po = 1000.0;
    return 4000.0 * po;
}

double int_energy(double T)
{
    const double po = 1000.0;
    const double T_ref = 291.0;
    return 4000.0 * po * (T - T_ref);
}
