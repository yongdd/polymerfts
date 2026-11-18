/*******************************************************************************
 * WARNING TO AI ASSISTANTS (Claude, ChatGPT, Copilot, etc.):
 * DO NOT MODIFY TEST PARAMETERS WITHOUT EXPLICIT PERMISSION FROM THE USER.
 * - NEVER increase tolerance values (e.g., 1e-7 -> 1e-6)
 * - NEVER decrease field strength or standard deviation values
 * - NEVER change grid sizes, box dimensions, or polymer parameters
 * - NEVER weaken any test conditions to make tests pass
 * These parameters are carefully calibrated. If a test fails, report the
 * failure to the user rather than modifying the test to pass.
 ******************************************************************************/

#include <iostream>
#include <cmath>
#include <numbers>
#include <complex>

#include "CudaComputationBox.h"

int main()
{
    try
    {
        CudaComputationBox<std::complex<double>>  cb({1,2,2}, {4.5,3.5,2.5}, {});
        double volume;
        std::complex<double> sum_w;
        std::complex<double> w[] 
            {{-0.8632002499891152, -0.2642949058151489},
            {-0.4400105438417454, 0.25515839348886193},
            {0.11957624566912606, -0.47053949095489656},
            {-0.47288841841362395, -0.989970480481541}};

        cuDoubleComplex g[]
            {{-0.9415024129393723, -0.49812524902468414},
            {0.41847784612200645, 0.8890323006002832},
            {-0.36140888131449245, -0.029759426767947073},
            {0.39470228341808133, -0.7504819613930107}};

        cuDoubleComplex h[] 
            {{0.32138498226345313, -0.030002215677094046},
            {0.5460116832979387, 0.12872199923910688},
            {0.4397920549860139, 0.44860363056686503},
            {0.6490069678310173, -0.6164591864981155}};

        cuDoubleComplex *d_g, *d_h;
        cudaMalloc((void**)&d_g, sizeof(cuDoubleComplex) * cb.get_total_grid());
        cudaMalloc((void**)&d_h, sizeof(cuDoubleComplex) * cb.get_total_grid());
        cudaMemcpy(d_g, g, sizeof(cuDoubleComplex) * cb.get_total_grid(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_h, h, sizeof(cuDoubleComplex) * cb.get_total_grid(), cudaMemcpyHostToDevice);

        volume = 0.0;
        for(int i=0; i<cb.get_total_grid(); i++)
            volume += cb.get_dv(i);
        
        cb.zero_mean(w);
        sum_w = 0.0;
        for(int i=0; i<cb.get_total_grid(); i++)
            sum_w += w[i];
        
        std::cout<< "nx: " << cb.get_nx(0) <<" "<< cb.get_nx(1) <<" "<< cb.get_nx(2) << std::endl;
        std::cout<< "lx: " << cb.get_lx(0) <<" "<< cb.get_lx(1) <<" "<< cb.get_lx(2) << std::endl;
        std::cout<< "dx: " << cb.get_dx(0) <<" "<< cb.get_dx(1) <<" "<< cb.get_dx(2) << std::endl;

        std::cout<< "sum(dV), Volume: " << volume <<" "<< cb.get_volume() << std::endl;
        std::cout<< "inner product: "<< cb.inner_product_device(d_g,d_h) << std::endl;
        std::cout<< "sum(w): " << sum_w << std::endl;
        
        if( !(std::abs(cb.inner_product_device(d_g,d_h)-std::complex<double>{-5.468649581714552, -4.903729566482727}) <= 1e-5))
            return -1;
        if( !(std::abs(sum_w) <= 1e-7))
            return -1;
        return 0;
    }
    catch(std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
