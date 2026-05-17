#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#define Ntr 1       //количество потоков
#define pi 3.14159
#define ln2 0.693147  // ln(2) for FWHM Gaussian
// Wavelength-dependent absorption (980 nm in water/agar)
#define lambda 980e-9   // Wavelength [m] (980 nm)
#define alpha_abs 48.2  // Absorption coefficient [1/m] at 980nm in water/agar ; изначально 48.2
// Grid dimensions adjusted for agar gel (larger domain, coarser mesh)
#define Nx1 102     //число узлов по оси х локальной сетки
#define Ny 150      //число узлов по оси у (increased for 8mm domain)
#define Nz 180      //число узлов по оси z (increased for 4mm depth)
#define Nx2 200     //число узлов по оси х глобальной сетки

/*функция для вычисления теплопроводности - AGAR GEL*/
double hcond(double T)
{
    double ks;
    ks = 0.6;  // Agar gel: ~0.6 W/(m·K), constant (was 6+0.015*(T-300) for titanium)
    return ks;
}

/*функция для вычисления теплоемкости - AGAR GEL*/
double hcap(double T)
{
    double Cs,
        po = 1000;  // Agar density: ~1000 kg/m³ (was 4300 for titanium)
    Cs = 4000 * po; // Agar: Cp = 4000 J/(kg·K), no phase transitions
    return Cs;
}

/*функция для вычисления приращения тепловой энергии*/
double int_energy(double T)
{
    double Eint,
           po = 1000,  // Agar density
           T0 = 291;
    Eint = 4000 * po * (T - T0);  // Simplified for agar (no phase transitions)
    return Eint;
}

int main(void)
{
    static double T1[Nx1][Ny][Nz], T2[Nx2][Ny][Nz], H1[Nx1][Ny], H2[Nx2][Ny];
    static double ATz1[Nz], BTz1[Nz], ATy1[Ny], BTy1[Ny], ATx1[Nx1], BTx1[Nx1];
    static double ATx2[Nx2], BTx2[Nx2], ATy2[Ny], BTy2[Ny], ATz2[Nz], BTz2[Nz];
    static double q0[Nx1][Ny];
    static double z[Nz], y[Ny];
    static double sum_z[Nx2][Ny], sum_zy[Nx2];
    double a, b, c, d, g, JT, Jn, jn, nominx, j1, j2, gn, uevmax,Tav,Esum,Eabs=0;
    double T0 = 291,
           pulse_fwhm = 3.0,    // Gaussian FWHM [s]
           pulse_sigma,         // Gaussian sigma = FWHM/2.355 [s]
           pulse_period,        // Period between pulses [s]
           t_peak,              // Time of peak in Gaussian
           r0 = 1.0e-3,         // Beam FWHM radius = 1 mm (FWHM diameter = 2 mm)
           l0 = 1.06e-6,
           A = 0.4,             // Absorptivity
           use_volumetric = 1,  // 1=Beer-Lambert volumetric absorption, 0=surface BC
           po=1000,
           Ep,                  // Energy per pulse [J]
           Q0,                  // Peak power density [W/m²]
           f,
           v,
           cs,
           uev,
           ks0,
           ks1;
    // Grid spacing: adjusted for 2mm beam
    double dx1 = r0/10;          // 100 μm for r0=1mm
    double dx2 = r0/5;           // 200 μm for r0=1mm
    double dt10, dt1, dt2;
    double time = 0, time1, time2, timestep = 10e-9, counttime;
    double delta = 0, Tmax = 0;
    FILE* f1, * f2, * f3, * f4, * f5, * f6;
    f1 = fopen("out_Txy.txt", "w");
    f2 = fopen("out_Tz.txt", "w");
    f3 = fopen("out_Tt.txt", "w");
    f4 = fopen("out_test.txt", "w");
    f5 = fopen("out_Txz.txt", "w");
    f6 = fopen("out_Tyz.txt", "w");
    int i, j, k, t, tmax1, tmax2, np, Np, np0;
    int i0, ind, i00 = Nx2/2;  // Center beam on surface (was 30)
    int ni;
    // Initial active domain (will expand adaptively)
    int Nzmax1 = 80;             // Increased for deeper domain
    int Nymax1 = 80;             // Increased from 40
    int Nxmax2 = i00 + 3 * (int)floor(r0 / dx2);
    int Nymax2 = 80;             // Increased from 40
    int Nzmax2 = 80;             // Increased for deeper domain


    /*ввод параметров*/
    printf("=======================================================\n");
    printf("  3 GAUSSIAN PULSES (FWHM = 3 s)\n");
    printf("=======================================================\n\n");
    printf("Input v (mm/s) [0 for stationary]: ");
    scanf("%lf", &v);
    printf("Input period between pulses (s): ");
    scanf("%lf", &pulse_period);
    printf("Input energy per pulse (J): ");
    scanf("%lf", &Ep);

    // Calculate Gaussian parameters
    // FWHM = 2*sqrt(2*ln(2)) * sigma = 2.355 * sigma
    pulse_sigma = pulse_fwhm / 2.355;  // sigma from FWHM
    t_peak = 3.0 * pulse_sigma;         // Peak at 3σ from start
    f = 1.0 / pulse_period;

    printf("\n=== Configuration ===\n");
    printf("Pulses: 3\n");
    printf("Pulse FWHM: %.1f s\n", pulse_fwhm);
    printf("Pulse sigma: %.3f s\n", pulse_sigma);
    printf("Period between pulses: %.1f s\n", pulse_period);
    printf("Energy/pulse: %.3e J\n", Ep);
    printf("Velocity: %.2f mm/s\n", v);
    printf("Beam FWHM diameter: %.1f mm (FWHM radius: %.1f mm)\n", r0 * 2e3, r0 * 1e3);
    printf("Beam position: center of surface\n");
    printf("Wavelength: %.0f nm\n", lambda * 1e9);
    printf("Absorption coefficient (alpha_abs): %.2f m^-1 (%.3f cm^-1)\n", alpha_abs, alpha_abs/100.0);
    printf("Penetration depth: %.1f mm\n", 1000.0 / alpha_abs);
    printf("Volumetric absorption: %s\n", use_volumetric ? "ENABLED (Beer-Lambert)" : "DISABLED (surface BC)");
    printf("=====================\n\n");


    clock_t timeprog;
    timeprog = clock();
    printf("executing...  %.2f%%\r", 0.0);
    fflush(stdout);
    v = v / 1e3;  // mm/s -> m/s
    i0 = i00;

    // Fixed: 3 pulses
    Np = 3;
    np0 = 1;  // Output at 2nd pulse

    // Uniform time stepping for entire simulation
    // For convergence: Fo = α·dt/dx² ≤ 0.5 → dt ≤ 3 ms for dx_min=30μm
    dt10 = 0.0005;  // 5 ms uniform time step (Fo ≈ 0.83, acceptable for ADI)
    dt1 = dt10;
    dt2 = dt10;

    // Peak power density from energy per pulse
    // For Gaussian: integral = Q0_peak * sqrt(2*pi) * sigma
    Q0 = Ep / (pi * pow(r0, 2) * sqrt(2.0 * pi) * pulse_sigma);

    /*генерация узлов сетки - адаптивная, независимая от r0*/
    // Y direction: 0 to ~10 mm with 150 points (lateral spread)
    y[0] = 0;
    for (j = 0; j < Ny - 1; j++)
    {
        // Start at 30 μm, grow to ~300 μm (coarser for faster computation)
        double dy = 30e-6 + pow(j + 1, 2.0) * 5e-7;
        if (dy > 300e-6) dy = 300e-6;
        y[j + 1] = y[j] + dy;
    }

    // Z direction: 0 to 5 mm depth with 180 points (FIXED DEPTH)
    z[0] = 0;
    for (k = 0; k < Nz - 1; k++)
    {
        // Start at 30 μm, grow to target 5 mm total depth (coarser for faster computation)
        double dz = 30e-6 + pow(k + 1, 1.8) * 2e-7;
        if (dz > 100e-6) dz = 100e-6;  // Cap at 100 μm
        z[k + 1] = z[k] + dz;
    }

    // Find z-indices for multiple depth measurements: 0mm, 1mm, 3mm
    int k_0mm = 0;  // Surface
    int k_1mm = 0;
    int k_3mm = 0;

    for (k = 0; k < Nz; k++) {
        if (z[k] >= 1e-3 && k_1mm == 0) k_1mm = k;
        if (z[k] >= 3e-3 && k_3mm == 0) k_3mm = k;
    }

    // Print domain size for verification
    printf("Grid generated:\n");
    printf("  Y: 0 to %.2f mm (%d points)\n", y[Ny-1]*1e3, Ny);
    printf("  Z: 0 to %.2f mm (%d points)\n", z[Nz-1]*1e3, Nz);
    printf("  Measurement points:\n");
    printf("    (0, 0, %.2f mm) at z-index %d\n", z[k_0mm]*1e3, k_0mm);
    printf("    (0, 0, %.2f mm) at z-index %d\n", z[k_1mm]*1e3, k_1mm);
    printf("    (0, 0, %.2f mm) at z-index %d\n", z[k_3mm]*1e3, k_3mm);

    /*задание начальных условий*/
    for (k = 0; k < Nz; k++)
        for (j = 0; j < Ny; j++)
        {
            for (i = 0; i < Nx1; i++)
              T1[i][j][k] = T0;
            for (i = 0; i < Nx2; i++)
              T2[i][j][k] = T0;
        }

   /*цикл по количенству импульсов*/
   for (np = 0; np < Np; np++)
    {
        time1 = 0;
        time2 = 0;
        ind = 1;

        /*расчет температуры в течение импульса - uniform dt*/
        while (time1 < 6.0 * pulse_sigma)
        {
#pragma omp parallel for shared(T1,Eabs) private(i,j,k,ATz1,BTz1,ks0,ks1,cs,a,b,c,d,g,JT) schedule (dynamic) num_threads(Ntr)
            for (i = 0; i < Nx1; i++)
                for (j = 0; j <= Nymax1; j++)
                {
                   // Gaussian temporal profile: exp(-(t-t_peak)²/(2σ²))
                   // Gaussian spatial profile: exp(-r²×ln(2)/r0²) where r0 is FWHM radius
                   q0[i][j] = A * Q0 * exp(-pow(time1 - t_peak, 2) / (2.0 * pow(pulse_sigma, 2))) *
                              exp(-(pow((i - Nx1 / 2) * dx1 - delta, 2) + pow(y[j], 2)) * ln2 / pow(r0, 2));

                   // Heat source at surface (z=0): volumetric or surface BC
                   if (use_volumetric == 0) {
                       // Old method: surface boundary condition (all energy at z=0)
                       JT = q0[i][j] / (z[1] - z[0]);
                   } else {
                       // New method: Beer-Lambert volumetric absorption
                       // Q(z) = α × I₀ × exp(-α×z) at z=0
                       JT = q0[i][j] * alpha_abs * exp(-alpha_abs * z[0]);
                   }
                   ks1 = hcond(T1[i][j][0]);
                   cs = hcap(T1[i][j][0]);
                   b = -(ks1 / pow(z[1] - z[0], 2) + cs / dt1);
                   c = ks1 / pow(z[1] - z[0], 2);
                   d = -(T1[i][j][0] * cs / dt1 + JT);
                   ATz1[0] = -c / b;
                   BTz1[0] = d / b;
                   for (k = 1; k < Nzmax1; k++)
                    {
                        // Volumetric heat source at depth z[k]
                        if (use_volumetric == 0) {
                            JT = 0;  // Old method: no volumetric source
                        } else {
                            // Beer-Lambert: Q(z) = α × I₀ × exp(-α×z)
                            JT = q0[i][j] * alpha_abs * exp(-alpha_abs * z[k]);
                        }
                        Jn = 0;
                        ks0 = hcond(T1[i][j][k]);
                        ks1 = hcond(T1[i][j][k + 1]);
                        cs = hcap(T1[i][j][k]);
                        a = ks0 / pow(z[k] - z[k - 1], 2);
                        b = -(cs / dt1 + ks0 / pow(z[k] - z[k - 1], 2) + ks1 / ((z[k + 1] - z[k]) * (z[k] - z[k - 1])));
                        c = ks1 / ((z[k + 1] - z[k]) * (z[k] - z[k - 1]));
                        d = -(T1[i][j][k] * cs / dt1 + JT);
                        g = a * ATz1[k - 1] + b;
                        ATz1[k] = -c / g;
                        BTz1[k] = (d - a * BTz1[k - 1]) / g;
                    }
                    ks1= hcond(T1[i][j][Nzmax1-1]);
                    cs= hcap(T1[i][j][Nzmax1-1]);
                    a = ks1 / pow(z[Nzmax1 - 1] - z[Nzmax1 - 2], 2);
                    b = -ks1 / pow(z[Nzmax1 - 1] - z[Nzmax1 - 2], 2) - cs / dt1;
                    d = -cs / dt1*T1[i][j][Nzmax1-1];
                    T1[i][j][Nzmax1 - 1] = (d - a * BTz1[Nzmax1 - 2]) / (b + a * ATz1[Nzmax1 - 2]);
                    for (k = Nzmax1 - 2; k >= 0; k--)
                      T1[i][j][k] = ATz1[k] * T1[i][j][k + 1] + BTz1[k];
                }
            if (r0<=10*sqrt(2.5*pulse_sigma*hcond(T0)/hcap(T0)))
            {
                for (i = 0; i < Nx1; i++)
                    for (k = 0; k < Nzmax1; k++)
                    {
                        JT = 0;
                        Jn = 0;
                        ks0 = hcond(T1[i][0][k]);
                        cs = hcap(T1[i][0][k]);
                        b = -(ks0 / pow(y[1] - y[0], 2) + cs / dt1);
                        c = ks0 / pow(y[1] - y[0], 2);
                        d = -(T1[i][0][k] * cs / dt1 + JT);
                        ATy1[0] = -c / b;
                        BTy1[0] = d / b;
                        for (j = 1; j < Nymax1; j++)
                        {
                            JT = 0;
                            Jn = 0;
                            ks0 = hcond(T1[i][j][k]);
                            ks1 = hcond(T1[i][j + 1][k]);
                            cs = hcap(T1[i][j][k]);
                            a = ks0 / pow(y[j] - y[j - 1], 2);
                            b = -(cs / dt1 + ks0 / pow(y[j] - y[j - 1], 2) + ks1 / ((y[j + 1] - y[j]) * (y[j] - y[j - 1])));
                            c = ks1 / ((y[j + 1] - y[j]) * (y[j] - y[j - 1]));
                            d = -(T1[i][j][k] * cs / dt1 + JT);
                            g = a * ATy1[j - 1] + b;
                            ATy1[j] = -c / g;
                            BTy1[j] = (d - a * BTy1[j - 1]) / g;
                        }
                        cs = hcap(T1[i][Nymax1][k]);
                        ks0 = hcond(T1[i][Nymax1][k]);
                        JT = 0;
                        Jn = 0;
                        a = ks0 / pow(y[Nymax1] - y[Nymax1 - 1], 2);
                        b = -(cs / dt1 + ks0 / pow(y[Nymax1] - y[Nymax1 - 1], 2));
                        d = -(T1[i][Nymax1][k] * cs / dt1 + JT);
                        T1[i][Nymax1][k] = (d - a * BTy1[Nymax1 - 1]) / (b + a * ATy1[Nymax1 - 1]);
                        for (j = Nymax1 - 1; j >= 0; j--)
                            T1[i][j][k] = ATy1[j] * T1[i][j + 1][k] + BTy1[j];
                    }
                for (k = 0; k < Nzmax1; k++)
                    for (j = 0; j <= Nymax1; j++)
                    {
                        JT = 0;
                        ks0 = hcond(T1[0][j][k]);
                        cs = hcap(T1[0][j][k]);
                        b = -(ks0 / pow(dx1, 2) + cs / dt1);
                        c = ks0 / pow(dx1, 2);
                        d = -(T1[0][j][k] * cs / dt1 + JT);
                        ATx1[0] = -c / b;
                        BTx1[0] = d / b;
                        for (i = 1; i < Nx1 - 1; i++)
                        {
                            JT = 0;
                            Jn = 0;
                            ks0 = hcond(T1[i][j][k]);
                            ks1 = hcond(T1[i + 1][j][k]);
                            cs = hcap(T1[i][j][k]);
                            a = ks0 / pow(dx1, 2);
                            b = -(cs / dt1 + (ks0 + ks1) / pow(dx1, 2));
                            c = ks1 / pow(dx1, 2);
                            d = -(T1[i][j][k] * cs / dt1 + JT);
                            g = a * ATx1[i - 1] + b;
                            ATx1[i] = -c / g;
                            BTx1[i] = (d - a * BTx1[i - 1]) / g;
                        }
                        cs = hcap(T1[Nx1 - 1][j][k]);
                        ks0 = hcond(T1[Nx1 - 1][j][k]);
                        JT = 0;
                        a = ks0 / pow(dx1, 2);
                        b = -(cs / dt1 + ks0 / pow(dx1, 2));
                        d = -(T1[Nx1 - 1][j][k] * cs / dt1 + JT);
                        T1[Nx1 - 1][j][k] = (d - a * BTx1[Nx1 - 2]) / (b + a * ATx1[Nx1 - 2]);
                        for (i = Nx1 - 2; i >= 0; i--)
                            T1[i][j][k] = ATx1[i] * T1[i + 1][j][k] + BTx1[i];
                    }
            }
            time1 = time1 + dt1;
            time = time + dt1;

            /*вывод данных в файл для зависиомсти температуры точки от времени*/
            // Output temperature at beam center at three depths: 0mm, 1mm, 3mm
            fprintf(f3, "%e  %.3f  %.3f  %.3f\n", time,
                    T1[Nx1/2][0][k_0mm], T1[Nx1/2][0][k_1mm], T1[Nx1/2][0][k_3mm]);
            counttime = counttime + dt1;


            /*определение момента времени достижения максимальной температуры в исследуемой точке и вывод данных в файлы*/
            if (np == np0)
                if (time1 > pulse_sigma)
                    if (ind == 1)
                        if (T1[Nx1 / 2][0][0] < Tmax)
                        {
                            for (i = 0; i < Nx1; i++)
                            {
                                for (j = 0; j < Ny; j++)
                                {
                                    fprintf(f1, "%e   %e   %.3f\n", ((i0 * dx2 - Nx1 / 2 * dx1) + i * dx1), y[j], T1[i][j][0]);
                                    if (j != 0)
                                        fprintf(f1, "%e   %e   %.3f\n", ((i0 * dx2 - Nx1 / 2 * dx1) + i * dx1), -y[j], T1[i][j][0]);
                                }
                                for (k = 0; k < Nz; k++)
                                    fprintf(f5, "%e   %e   %.3f\n", ((i0 * dx2 - Nx1 / 2 * dx1) + i * dx1), z[k], T1[i][0][k]);

                            }
                            for (i = 0; i < Nx2; i++)
                            {
                                for (j = 0; j < Ny; j++)
                                {
                                    if (i * dx2<(i0 * dx2 - Nx1 / 2 * dx1) || i * dx2>(i0 * dx2 + Nx1 / 2 * dx1))
                                    {
                                        fprintf(f1, "%e   %e   %.3f\n", i * dx2, y[j], T2[i][j][0]);
                                        if (j != 0)
                                            fprintf(f1, "%e   %e   %.3f\n", i * dx2, -y[j], T2[i][j][0]);
                                    }
                                }

                                for (k = 0; k < Nz; k++)
                                    if (i * dx2<(i0 * dx2 - Nx1 / 2 * dx1) || i * dx2>(i0 * dx2 + Nx1 / 2 * dx1))
                                     fprintf(f5, "%e   %e   %.3f\n", i * dx2, z[k], T2[i][0][k]);

                            }
                            for(k=0;k<Nz;k++)
                                for (j = 0; j < Ny; j++)
                                {
                                    fprintf(f6, "%e   %e   %.3f\n", y[j], z[k], T1[Nx1/2][j][k]);
                                    if (j != 0)
                                        fprintf(f6, "%e   %e   %.3f\n", -y[j], z[k], T1[Nx1/2][j][k]);
                                }


                                ind = 0;
                        }
         Tmax = T1[Nx1 / 2][0][0];
        }


        /*перерасчет температуры на глобальной сетке*/
        for (j = 0; j < Ny; j++)
            for (k = 0; k < Nz; k++)
                for (i = i0; i < Nx2; i++)
                {
                    if (3 * (i - i0) < Nx1/2)
                    {
                        T2[i][j][k] = T1[Nx1 / 2 + 3 * (i - i0)][j][k];
                        T2[2 * i0 - 1 - i][j][k] = T1[Nx1 / 2 - 3 * (i + 1 - i0)][j][k];
                    }

                }
        dt2 = dt1;

        /*расчет на стадии остывания между импульсами - uniform dt*/
        while (time2 < (1 / f - time1))
        {
            // Use uniform dt (no adaptive stepping)
#pragma omp parallel for shared(T2) private(i,j,k,ATz2,BTz2,ks0,ks1,cs,a,b,c,d,g,JT) schedule (dynamic) num_threads(Ntr)
            for (i = 0; i < Nxmax2; i++)
                for (j = 0; j < Nymax2; j++)
                {
                    JT = 0;
                    ATz2[0] = 1;
                    BTz2[0] = 0;
                    for (k = 1; k < Nzmax2 - 1; k++)
                    {
                        JT = 0;
                        ks0 = hcond(T2[i][j][k]);
                        ks1 = hcond(T2[i][j][k + 1]);
                        cs = hcap(T2[i][j][k]);
                        a = ks0 / pow(z[k] - z[k - 1], 2);
                        b = -(cs / dt2 + ks0 / pow(z[k] - z[k - 1], 2) + ks1 / ((z[k + 1] - z[k]) * (z[k] - z[k - 1])));
                        c = ks1 / ((z[k + 1] - z[k]) * (z[k] - z[k - 1]));
                        d = -(T2[i][j][k] * cs / dt2 + JT);
                        g = a * ATz2[k - 1] + b;
                        ATz2[k] = -c / g;
                        BTz2[k] = (d - a * BTz2[k - 1]) / g;
                    }
                    ks1 = hcond(T2[i][j][Nzmax2 - 1]);
                    cs = hcap(T2[i][j][Nzmax2 - 1]);
                    a = ks1 / pow(z[Nzmax2 - 1] - z[Nzmax2 - 2], 2);
                    b = -ks1 / pow(z[Nzmax2 - 1] - z[Nzmax2 - 2], 2) - cs / dt2;
                    d = -cs / dt2 * T2[i][j][Nzmax2 - 1];
                    T2[i][j][Nzmax2 - 1] = (d - a * BTz2[Nzmax2 - 2]) / (b + a * ATz2[Nzmax2 - 2]);
                    for (k = Nzmax2 - 2; k >= 0; k--)
                      T2[i][j][k] = ATz2[k] * T2[i][j][k + 1] + BTz2[k];
                }
#pragma omp parallel for shared(T2) private(i,j,k,ATy2,BTy2,ks0,ks1,cs,a,b,c,d,g,JT) schedule (dynamic) num_threads(Ntr)
                for (i = 0; i < Nxmax2; i++)
                 for (k = 0; k < Nzmax2; k++)
                 {
                    JT = 0;
                    Jn = 0;
                    ATy2[0] = 1;
                    BTy2[0] = 0;
                    for (j = 1; j < Nymax2 - 1; j++)
                    {
                        JT = 0;
                        Jn = 0;
                        ks0 = hcond(T2[i][j][k]);
                        ks1 = hcond(T2[i][j + 1][k]);
                        cs = hcap(T2[i][j][k]);
                        a = ks0 / pow(y[j] - y[j - 1], 2);
                        b = -(cs / dt2 + ks0 / pow(y[j] - y[j - 1], 2) + ks1 / ((y[j + 1] - y[j]) * (y[j] - y[j - 1])));
                        c = ks1 / ((y[j + 1] - y[j]) * (y[j] - y[j - 1]));
                        d = -(T2[i][j][k] * cs / dt2 + JT);
                        g = a * ATy2[j - 1] + b;
                        ATy2[j] = -c / g;
                        BTy2[j] = (d - a * BTy2[j - 1]) / g;

                    }
                    ks1 = hcond(T2[i][Nymax2 - 1][k]);
                    cs = hcap(T2[i][Nymax2 - 1][k]);
                    a = ks1 / pow(y[Nymax2 - 1] - y[Nymax2 - 2], 2);
                    b = -ks1 / pow(y[Nymax2 - 1] - y[Nymax2 - 2], 2) - cs / dt2;
                    d = -cs / dt2 * T2[i][Nymax2 - 1][k];
                    T2[i][Nymax2 - 1][k] = (d - a * BTy2[Nymax2 - 2]) / (b + a * ATy2[Nymax2 - 2]);
                    for (j = Nymax2 - 2; j >= 0; j--)
                     T2[i][j][k] = ATy2[j] * T2[i][j + 1][k] + BTy2[j];

                 }
#pragma omp parallel for shared(T2) private(i,j,k,ATx2,BTx2,ks0,ks1,cs,a,b,c,d,g,JT) schedule (dynamic) num_threads(Ntr)
            for (j = 0; j < Nymax2; j++)
                for (k = 0; k < Nzmax2; k++)
                {
                    JT = 0;
                    ATx2[0] = 1;
                    BTx2[0] = 0;
                    for (i = 1; i < Nxmax2 - 1; i++)
                    {
                        JT = 0;

                        ks0 = hcond(T2[i][j][k]);
                        ks1 = hcond(T2[i + 1][j][k]);
                        cs = hcap(T2[i][j][k]);
                        a = ks0 / pow(dx2, 2);
                        b = -(cs / dt2 + (ks0 + ks1) / pow(dx2, 2));
                        c = ks1 / pow(dx2, 2);
                        d = -(T2[i][j][k] * cs / dt2 + JT);
                        g = a * ATx2[i - 1] + b;
                        ATx2[i] = -c / g;
                        BTx2[i] = (d - a * BTx2[i - 1]) / g;

                    }
                    ks1 = hcond(T2[Nxmax2 - 1][j][k]);
                    cs = hcap(T2[Nxmax2 - 1][j][k]);
                    a = ks1 / pow(dx2, 2);
                    b = -ks1 / pow(dx2, 2) - cs / dt2;
                    d = -cs / dt2 * T2[Nxmax2 - 1][j][k];
                    T2[Nxmax2 - 1][j][k] = (d - a * BTx2[Nxmax2 - 2]) / (b + a * ATx2[Nxmax2 - 2]);
                    for (i = Nxmax2 - 2; i >= 0; i--)
                      T2[i][j][k] = ATx2[i] * T2[i + 1][j][k] + BTx2[i];
                }
            time2 = time2 + dt2;
            time = time + dt2;
            // Output temperature at beam center at three depths: 0mm, 1mm, 3mm
            fprintf(f3, "%e  %.3f  %.3f  %.3f\n", time,
                    T2[i0][0][k_0mm], T2[i0][0][k_1mm], T2[i0][0][k_3mm]);

          /*проверка сохранения энергии*/
           /*Esum = 0;
             for(i=0;i<Nx2-1;i++)
                for(j=0;j<Ny-1;j++)
                    for (k = 0; k < Nz - 1; k++)
                    {
                        Tav = (T2[i][j][k] + T2[i][j][k + 1] + T2[i][j + 1][k] + T2[i][j + 1][k + 1] + T2[i + 1][j][k] + T2[i + 1][j][k + 1] + T2[i + 1][j + 1][k] + T2[i + 1][j + 1][k + 1])/8;
                        Esum = Esum + int_energy(Tav) * dx2 * (y[j + 1] - y[j]) * (z[k + 1] - z[k]);
                    }*/
           // printf("%e  %e  %e  %e\n",time, A * P / f * (np + 1), 2 * Esum, 2*Eabs);
           //fprintf(f4, "%e  %e  %e  %e  %e  %e\n", time, A* P / f * (np + 1), 2 * Esum,fabs(2 * Esum - A * P / f * (np + 1)) / (A * P / f * (np + 1)));
           if (T2[Nxmax2 - 1][0][0] > T0 + 1)
                Nxmax2 = Nxmax2 + 1;
            if (T2[i00][Nymax2 - 1][0] > T0 + 1)
                Nymax2 = Nymax2 + 1;
            if (T2[i00][0][Nzmax2-1] > T0 + 1)
                Nzmax2 = Nzmax2 + 1;
            if (Nxmax2 > Nx2)
                Nxmax2 = Nx2;
            if (Nymax2 > Ny)
                Nymax2 = Ny;
            if (Nzmax2 > Nz)
                Nzmax2 = Nz;
        }
        i0 = i00 + (np + 1) * v / f / dx2;
        delta = i00 * dx2 + (np + 1) * v / f - i0 * dx2;
        if ((i0 * dx2 + 3 * r0) > Nxmax2 * dx2)
            Nxmax2 = i0 + 3 * (int)floor(r0 / dx2);
        if (Nxmax2 > Nx2)
            Nxmax2 = Nx2;

        /*перерасчет температуры на локальную сетку*/
        for (i = Nx1 / 2; i < Nx1; i++)

            for (j = 0; j < Ny; j++)
                for (k = 0; k < Nz; k++)
                {
                    ni = i0 + (i - Nx1 / 2) / 3;
                    T1[i][j][k] = T2[ni][j][k] + (dx1 * (i - Nx1 / 2) - (ni - i0) * dx2) * (T2[ni + 1][j][k] - T2[ni][j][k]) / dx2;
                    ni = i0 - (i - Nx1 / 2) / 3 - 1;
                    T1[Nx1 - i - 1][j][k] = T2[ni][j][k] + ((i0 - ni) * dx2 - (i - Nx1 / 2 + 1) * dx1) * (T2[ni + 1][j][k] - T2[ni][j][k]) / dx2;
                }
        printf("executing...  %.2f%%\r", 100 * (double)(np + 1) / Np);
        fflush(stdout);

    }

    // Extended cooling: last pulse ends at (2×period + 6σ), then add 10s
    double target_time = 2.0 * pulse_period + 6.0 * pulse_sigma + 10.0;  // Total simulation time [s]
    printf("\nExtended cooling to %.1f s (end of pulse 3 + 10s)...\n", target_time);

    while (time < target_time)
    {
        // ADI solve on global grid T2 (same as cooling phase)
#pragma omp parallel for shared(T2) private(i,j,k,ATz2,BTz2,ks0,ks1,cs,a,b,c,d,g,JT) schedule (dynamic) num_threads(Ntr)
        for (i = 0; i < Nxmax2; i++)
            for (j = 0; j < Nymax2; j++)
            {
                JT = 0;
                ATz2[0] = 1;
                BTz2[0] = 0;
                for (k = 1; k < Nzmax2 - 1; k++)
                {
                    JT = 0;
                    ks0 = hcond(T2[i][j][k]);
                    ks1 = hcond(T2[i][j][k + 1]);
                    cs = hcap(T2[i][j][k]);
                    a = ks0 / pow(z[k] - z[k - 1], 2);
                    b = -(cs / dt2 + ks0 / pow(z[k] - z[k - 1], 2) + ks1 / ((z[k + 1] - z[k]) * (z[k] - z[k - 1])));
                    c = ks1 / ((z[k + 1] - z[k]) * (z[k] - z[k - 1]));
                    d = -(T2[i][j][k] * cs / dt2 + JT);
                    g = a * ATz2[k - 1] + b;
                    ATz2[k] = -c / g;
                    BTz2[k] = (d - a * BTz2[k - 1]) / g;
                }
                ks1 = hcond(T2[i][j][Nzmax2 - 1]);
                cs = hcap(T2[i][j][Nzmax2 - 1]);
                a = ks1 / pow(z[Nzmax2 - 1] - z[Nzmax2 - 2], 2);
                b = -ks1 / pow(z[Nzmax2 - 1] - z[Nzmax2 - 2], 2) - cs / dt2;
                d = -cs / dt2 * T2[i][j][Nzmax2 - 1];
                T2[i][j][Nzmax2 - 1] = (d - a * BTz2[Nzmax2 - 2]) / (b + a * ATz2[Nzmax2 - 2]);
                for (k = Nzmax2 - 2; k >= 0; k--)
                  T2[i][j][k] = ATz2[k] * T2[i][j][k + 1] + BTz2[k];
            }
#pragma omp parallel for shared(T2) private(i,j,k,ATy2,BTy2,ks0,ks1,cs,a,b,c,d,g,JT) schedule (dynamic) num_threads(Ntr)
        for (i = 0; i < Nxmax2; i++)
         for (k = 0; k < Nzmax2; k++)
         {
            JT = 0;
            Jn = 0;
            ATy2[0] = 1;
            BTy2[0] = 0;
            for (j = 1; j < Nymax2 - 1; j++)
            {
                JT = 0;
                Jn = 0;
                ks0 = hcond(T2[i][j][k]);
                ks1 = hcond(T2[i][j + 1][k]);
                cs = hcap(T2[i][j][k]);
                a = ks0 / pow(y[j] - y[j - 1], 2);
                b = -(cs / dt2 + ks0 / pow(y[j] - y[j - 1], 2) + ks1 / ((y[j + 1] - y[j]) * (y[j] - y[j - 1])));
                c = ks1 / ((y[j + 1] - y[j]) * (y[j] - y[j - 1]));
                d = -(T2[i][j][k] * cs / dt2 + JT);
                g = a * ATy2[j - 1] + b;
                ATy2[j] = -c / g;
                BTy2[j] = (d - a * BTy2[j - 1]) / g;

            }
            ks1 = hcond(T2[i][Nymax2 - 1][k]);
            cs = hcap(T2[i][Nymax2 - 1][k]);
            a = ks1 / pow(y[Nymax2 - 1] - y[Nymax2 - 2], 2);
            b = -ks1 / pow(y[Nymax2 - 1] - y[Nymax2 - 2], 2) - cs / dt2;
            d = -cs / dt2 * T2[i][Nymax2 - 1][k];
            T2[i][Nymax2 - 1][k] = (d - a * BTy2[Nymax2 - 2]) / (b + a * ATy2[Nymax2 - 2]);
            for (j = Nymax2 - 2; j >= 0; j--)
                T2[i][j][k] = ATy2[j] * T2[i][j + 1][k] + BTy2[j];

        }
#pragma omp parallel for shared(T2) private(i,j,k,ATx2,BTx2,ks0,ks1,cs,a,b,c,d,g,JT) schedule (dynamic) num_threads(Ntr)
        for (j = 0; j < Nymax2; j++)
            for (k = 0; k < Nzmax2; k++)
            {
                JT = 0;
                ATx2[0] = 1;
                BTx2[0] = 0;
                for (i = 1; i < Nxmax2 - 1; i++)
                {
                    JT = 0;
                    ks0 = hcond(T2[i][j][k]);
                    ks1 = hcond(T2[i + 1][j][k]);
                    cs = hcap(T2[i][j][k]);
                    a = ks0 / pow(dx2, 2);
                    b = -(cs / dt2 + 2 * ks1 / pow(dx2, 2));
                    c = ks1 / pow(dx2, 2);
                    d = -(T2[i][j][k] * cs / dt2 + JT);
                    g = a * ATx2[i - 1] + b;
                    ATx2[i] = -c / g;
                    BTx2[i] = (d - a * BTx2[i - 1]) / g;

                }
                ks1 = hcond(T2[Nxmax2 - 1][j][k]);
                cs = hcap(T2[Nxmax2 - 1][j][k]);
                a = ks1 / pow(dx2, 2);
                b = -ks1 / pow(dx2, 2) - cs / dt2;
                d = -cs / dt2 * T2[Nxmax2 - 1][j][k];
                T2[Nxmax2 - 1][j][k] = (d - a * BTx2[Nxmax2 - 2]) / (b + a * ATx2[Nxmax2 - 2]);
                for (i = Nxmax2 - 2; i >= 0; i--)
                  T2[i][j][k] = ATx2[i] * T2[i + 1][j][k] + BTx2[i];
            }
        time = time + dt2;
        // Output temperature at beam center at three depths: 0mm, 1mm, 3mm
        fprintf(f3, "%e  %.3f  %.3f  %.3f\n", time,
                T2[i0][0][k_0mm], T2[i0][0][k_1mm], T2[i0][0][k_3mm]);

        if (T2[Nxmax2 - 1][0][0] > T0 + 1)
            Nxmax2 = Nxmax2 + 1;
    }
    printf("Extended cooling complete. Final time: %.1f s\n", time);

    fclose(f1);
    fclose(f2);
    fclose(f3);
    fclose(f4);
    fclose(f5);
    fclose(f6);

    timeprog = clock() - timeprog;
    printf("\nProgram execution time=%f s\n", (double)timeprog / CLOCKS_PER_SEC);
    printf("Press Enter to continue...");
    getchar();
    return 0;

}


