#include <iostream>
#include <cmath>
#include <numbers>
#include <algorithm>
#include <array>
#include <vector>
#include <CudaCommon.h>
#include "CudaSolverCNADI.h"
#include "TestNanSafe.h"

int main()
{
    try
    {
        const int M = 10;
        double a[M] = {0.822383458999126,  0.180877073118435, 0.885692320279145,
                       0.89417010799111,   0.495074990864166, 0.612975741629382,
                       0.0415795198090432, 0.353431810889399, 0.773118461366249,
                       0.474587294381635};

        double b[M] = {0.200002276650706, 0.592127025743285, 0.460207620078036,
                       0.435945198378862, 0.61269588805607,  0.355979618324841,
                       0.548759176402544, 0.482897565408353, 0.541501788021353,
                       0.349682106604464};

        double c[M] = {0.104943438436657, 0.418086769592863, 0.61190542613784,
                       0.792961240687622, 0.713098832553561, 0.667410867822433,
                       0.492427261460169, 0.261956970404376, 0.479635996452285,
                       0.206215439022739};

        double y[M] = {0.180066106322814, 0.349989269840229, 0.580249533529743,
                       0.653060847246207, 0.793729416310673, 0.988032605316576,
                       0.98005550969782,  0.38678227079795,  0.894395839154923,
                       0.720491484453521};

        double x_answer[M] = {1.30521858221904, -0.771659313380314,  1.36532779447559,
                              1.0383434746621,  -1.28686656279112,   1.49787004664211,
                              1.86337898208832, -0.212765480943989, -0.645339592235395,
                              2.93627107620712};
        double x_periodic_answer[M] = {-37.7095597725768, -8.90673671271047, 29.7658392801646,
                                       -8.54639948044212, -28.0428600017516, 31.1409611959355,
                                        10.6262549404399, -12.4810831844606, 10.1475079039034,
                                        10.526436777743};

        double x[M] = {0.0};
        std::array<double,M> diff_sq;
        double error;

        // The solvers sweep over the factors of the matrix (lower a, diagonal b, upper c),
        // computed once on the host
        const TridiagonalFactors factors = factorize_tridiagonal(a, b, c, M, false);
        const TridiagonalFactors factors_periodic = factorize_tridiagonal(a, b, c, M, true);
        CudaDeviceBuffer<double> d_table(4*M), d_table_periodic(4*M), d_y(M), d_x(M);
        CudaDeviceBuffer<int> d_offset(1);
        const int offset = 0;
        CUDA_CHECK(cudaMemcpy(d_table, factors.table.data(), sizeof(double)*4*M, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_table_periodic, factors_periodic.table.data(), sizeof(double)*4*M, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y, y, sizeof(double)*M, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_offset, &offset, sizeof(int), cudaMemcpyHostToDevice));
        // One line in shared memory: a row of M + 1 values (M is even)
        const size_t shmem_size = (M + 1) * sizeof(double);

        // Tridiagonal
        tridiagonal_lines_shared<false><<<1, 1, shmem_size>>>(d_table, M, factors.v, factors.denom, d_y, d_x, 1);
        CUDA_CHECK_LAUNCH();
        CUDA_CHECK(cudaMemcpy(x, d_x, sizeof(double)*M, cudaMemcpyDeviceToHost));
        std::cout << "Tridiagonal" << std::endl;
        std::cout << "i: x_answer[i], x[i]" << std::endl;
        for(int i=0; i<M; i++)
            std::cout << i << ": " << x_answer[i] << ", " << x[i] << std::endl;
        for(int i=0; i<M; i++)
            diff_sq[i] = pow(x_answer[i] - x[i],2);
        error = sqrt(max_or_nan(diff_sq));

        std::cout<< "Inverse error: "<< error << std::endl;
        if (!std::isfinite(error) || error > 1e-7)
            return -1;

        // Tridiagonal with periodic boundary
        tridiagonal_lines_shared<true><<<1, 1, shmem_size>>>(d_table_periodic, M, factors_periodic.v,
                                                              factors_periodic.denom, d_y, d_x, 1);
        CUDA_CHECK_LAUNCH();
        CUDA_CHECK(cudaMemcpy(x, d_x, sizeof(double)*M, cudaMemcpyDeviceToHost));
        std::cout << "Tridiagonal with periodic boundary" << std::endl;
        std::cout << "i: x_periodic_answer[i], x[i]" << std::endl;
        for(int i=0; i<M; i++)
            std::cout << i << ": " << x_periodic_answer[i] << ", " << x[i] << std::endl;
        for(int i=0; i<M; i++)
            diff_sq[i] = pow(x_periodic_answer[i] - x[i],2);
        error = sqrt(max_or_nan(diff_sq));

        std::cout<< "Inverse error: "<< error << std::endl;
        if (!std::isfinite(error) || error > 1e-7)
            return -1;

        // One line per thread in global memory (used along strided axes and for lines too long
        // for shared memory): the same arithmetic as through shared memory
        for (bool periodic : {false, true})
        {
            const TridiagonalFactors& f = periodic ? factors_periodic : factors;
            const double* table = periodic ? d_table_periodic.get() : d_table.get();
            std::array<double, M> x_shared;
            if (periodic)
                tridiagonal_lines_shared<true><<<1, 1, shmem_size>>>(table, M, f.v, f.denom, d_y, d_x, 1);
            else
                tridiagonal_lines_shared<false><<<1, 1, shmem_size>>>(table, M, f.v, f.denom, d_y, d_x, 1);
            CUDA_CHECK_LAUNCH();
            CUDA_CHECK(cudaMemcpy(x_shared.data(), d_x, sizeof(double)*M, cudaMemcpyDeviceToHost));
            if (periodic)
                tridiagonal_lines<true><<<1, 64>>>(table, M, f.v, f.denom, d_y, d_x, d_offset, 1, 1);
            else
                tridiagonal_lines<false><<<1, 64>>>(table, M, f.v, f.denom, d_y, d_x, d_offset, 1, 1);
            CUDA_CHECK_LAUNCH();
            CUDA_CHECK(cudaMemcpy(x, d_x, sizeof(double)*M, cudaMemcpyDeviceToHost));
            std::cout << "Tridiagonal from global memory" << (periodic ? ", periodic" : "") << std::endl;
            if (!std::equal(x_shared.begin(), x_shared.end(), x))
            {
                std::cout << "tridiagonal_lines differs from tridiagonal_lines_shared" << std::endl;
                return -1;
            }
        }

        // A long line, through global memory and through shared memory
        const int L = 6000;
        std::vector<double> la(L), lb(L), lc(L), ly(L), lx(L);
        for (int i = 0; i < L; i++)
        {
            la[i] = -0.5 - 0.1*std::sin(1.0*i);
            lb[i] =  2.5 + 0.2*std::cos(0.3*i);
            lc[i] = -0.6 + 0.1*std::sin(0.7*i);
            ly[i] = std::cos(0.01*i);
        }
        const TridiagonalFactors long_factors = factorize_tridiagonal(la.data(), lb.data(), lc.data(), L, false);
        CudaDeviceBuffer<double> d_long_table(4*L), d_ly(L), d_lx(L);
        CUDA_CHECK(cudaMemcpy(d_long_table, long_factors.table.data(), sizeof(double)*4*L, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ly, ly.data(), sizeof(double)*L, cudaMemcpyHostToDevice));
        for (bool shared : {false, true})
        {
            if (shared)
                tridiagonal_lines_shared<false><<<1, 1, (L + 1)*sizeof(double)>>>(
                    d_long_table, L, long_factors.v, long_factors.denom, d_ly, d_lx, 1);
            else
                tridiagonal_lines<false><<<1, 64>>>(d_long_table, L, long_factors.v, long_factors.denom,
                                                    d_ly, d_lx, d_offset, 1, 1);
            CUDA_CHECK_LAUNCH();
            CUDA_CHECK(cudaMemcpy(lx.data(), d_lx, sizeof(double)*L, cudaMemcpyDeviceToHost));
            std::vector<double> residual(L);
            for (int i = 0; i < L; i++)
            {
                double r = lb[i]*lx[i] - ly[i];
                if (i > 0)
                    r += la[i]*lx[i-1];
                if (i < L-1)
                    r += lc[i]*lx[i+1];
                residual[i] = std::abs(r);
            }
            error = max_or_nan(residual);
            std::cout << "Tridiagonal of " << L << " points from " << (shared ? "shared" : "global")
                      << " memory, max residual: " << error << std::endl;
            if (!std::isfinite(error) || error > 1e-12)
                return -1;
        }

        return 0;
    }
    catch(std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
