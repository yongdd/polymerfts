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

#include <cstdlib>
#include <iostream>
#include <cmath>
#include <numbers>
#include <algorithm>

#include "Exception.h"
#include "Polymer.h"
#include "PropagatorComputationOptimizer.h"
#include "TestComputationFactory.h"
#include "TestNanSafe.h"
#include "ComputationBox.h"
#ifdef USE_CUDA
#include "CudaComputationBox.h"
#include <cuda_runtime.h>
// A CUDA build tested on a node without a visible GPU fails, like the TestCuda*
// tests: a skipped CUDA part would read as a passed CUDA test.
[[maybe_unused]] static bool cuda_device_visible()
{
    int n_devices = 0;
    if (cudaGetDeviceCount(&n_devices) != cudaSuccess || n_devices == 0)
    {
        std::cout << "FAILED: this build has the CUDA backend, but no CUDA device is visible." << std::endl;
        std::exit(1);
    }
    return true;
}
#endif

int main()
{
    try
    {
        const int II{10};
        const int M{II};
        const int NN{4};

        double q_next[M];

        std::array<double,M> diff_sq;
        double error;
        double Lx, f;

        f = 0.5;
        Lx = 4.0;

        double w_a[M] = {0.822383458999126,  0.180877073118435, 0.885692320279145,
                       0.89417010799111,   0.495074990864166, 0.612975741629382,
                       0.0415795198090432, 0.353431810889399, 0.773118461366249,
                       0.474587294381635};

        double w_b[M] = {0.200002276650706, 0.592127025743285, 0.460207620078036,
                       0.435945198378862, 0.61269588805607,  0.355979618324841,
                       0.548759176402544, 0.482897565408353, 0.541501788021353,
                       0.349682106604464};

        //-------------- initialize ------------
        std::cout<< "Initializing" << std::endl;
        std::map<std::string, double> bond_lengths = {{"A",1.0}, {"B",1.0}};
        std::vector<BlockInput> blocks = 
        {
            {"A",    f, 0, 1},
            {"B",1.0-f, 1, 2},
        };

        std::vector<std::string> bc_abs = 
        {
            "absorbing", "absorbing",
        };
        std::vector<std::string> bc_rfl = 
        {
            "reflecting", "reflecting",
        };
        std::vector<std::string> bc_prd = 
        {
            "periodic", "periodic",
        };

        Molecules* molecules = new Molecules("Continuous", 1.0/NN, bond_lengths);
        molecules->add_polymer(1.0, blocks, {});
        PropagatorComputationOptimizer* propagator_computation_optimizer= new PropagatorComputationOptimizer(molecules, false);

        propagator_computation_optimizer->display_blocks();
        propagator_computation_optimizer->display_propagators();

        std::vector<PropagatorComputation<double>*> solver_list;
        std::vector<std::string> solver_name_list;

        int repeat = 0;

        #ifdef USE_CPU_FFTW
        repeat += 2;
        solver_name_list.push_back("cpu-fftw, absorbing");
        solver_name_list.push_back("cpu-fftw, absorbing, reduce_memory");
        solver_list.push_back(make_propagator_computation<double>("cpu-fftw", false, new ComputationBox<double>({II}, {Lx}, bc_abs), molecules, propagator_computation_optimizer, "cn-adi2"));
        solver_list.push_back(make_propagator_computation<double>("cpu-fftw", true, new ComputationBox<double>({II}, {Lx}, bc_abs), molecules, propagator_computation_optimizer, "cn-adi2"));
        #endif

        #ifdef USE_CPU_MKL
        repeat += 2;
        solver_name_list.push_back("cpu-mkl, absorbing");
        solver_name_list.push_back("cpu-mkl, absorbing, reduce_memory");
        solver_list.push_back(make_propagator_computation<double>("cpu-mkl", false, new ComputationBox<double>({II}, {Lx}, bc_abs), molecules, propagator_computation_optimizer, "cn-adi2"));
        solver_list.push_back(make_propagator_computation<double>("cpu-mkl", true, new ComputationBox<double>({II}, {Lx}, bc_abs), molecules, propagator_computation_optimizer, "cn-adi2"));
        #endif

        #ifdef USE_CUDA
        if (cuda_device_visible())
        {
            repeat += 2;
            solver_name_list.push_back("cuda, absorbing");
            solver_name_list.push_back("cuda, absorbing, reduce_memory");
            solver_list.push_back(make_propagator_computation<double>("cuda", false, new CudaComputationBox<double>({II}, {Lx}, bc_abs), molecules, propagator_computation_optimizer, "cn-adi2"));
            solver_list.push_back(make_propagator_computation<double>("cuda", true, new CudaComputationBox<double>({II}, {Lx}, bc_abs), molecules, propagator_computation_optimizer, "cn-adi2"));
        }
        #endif

        #ifdef USE_CPU_FFTW
        solver_name_list.push_back("cpu-fftw, reflecting");
        solver_name_list.push_back("cpu-fftw, reflecting, reduce_memory");
        solver_list.push_back(make_propagator_computation<double>("cpu-fftw", false, new ComputationBox<double>({II}, {Lx}, bc_rfl), molecules, propagator_computation_optimizer, "cn-adi2"));
        solver_list.push_back(make_propagator_computation<double>("cpu-fftw", true, new ComputationBox<double>({II}, {Lx}, bc_rfl), molecules, propagator_computation_optimizer, "cn-adi2"));
        #endif

        #ifdef USE_CPU_MKL
        solver_name_list.push_back("cpu-mkl, reflecting");
        solver_name_list.push_back("cpu-mkl, reflecting, reduce_memory");
        solver_list.push_back(make_propagator_computation<double>("cpu-mkl", false, new ComputationBox<double>({II}, {Lx}, bc_rfl), molecules, propagator_computation_optimizer, "cn-adi2"));
        solver_list.push_back(make_propagator_computation<double>("cpu-mkl", true, new ComputationBox<double>({II}, {Lx}, bc_rfl), molecules, propagator_computation_optimizer, "cn-adi2"));
        #endif

        #ifdef USE_CUDA
        if (cuda_device_visible())
        {
            solver_name_list.push_back("cuda, reflecting");
            solver_name_list.push_back("cuda, reflecting, reduce_memory");
            solver_list.push_back(make_propagator_computation<double>("cuda", false, new CudaComputationBox<double>({II}, {Lx}, bc_rfl), molecules, propagator_computation_optimizer, "cn-adi2"));
            solver_list.push_back(make_propagator_computation<double>("cuda", true, new CudaComputationBox<double>({II}, {Lx}, bc_rfl), molecules, propagator_computation_optimizer, "cn-adi2"));
        }
        #endif

        #ifdef USE_CPU_FFTW
        solver_name_list.push_back("cpu-fftw, periodic");
        solver_name_list.push_back("cpu-fftw, periodic, reduce_memory");
        solver_list.push_back(make_propagator_computation<double>("cpu-fftw", false, new ComputationBox<double>({II}, {Lx}, bc_prd), molecules, propagator_computation_optimizer, "cn-adi2"));
        solver_list.push_back(make_propagator_computation<double>("cpu-fftw", true, new ComputationBox<double>({II}, {Lx}, bc_prd), molecules, propagator_computation_optimizer, "cn-adi2"));
        #endif

        #ifdef USE_CPU_MKL
        solver_name_list.push_back("cpu-mkl, periodic");
        solver_name_list.push_back("cpu-mkl, periodic, reduce_memory");
        solver_list.push_back(make_propagator_computation<double>("cpu-mkl", false, new ComputationBox<double>({II}, {Lx}, bc_prd), molecules, propagator_computation_optimizer, "cn-adi2"));
        solver_list.push_back(make_propagator_computation<double>("cpu-mkl", true, new ComputationBox<double>({II}, {Lx}, bc_prd), molecules, propagator_computation_optimizer, "cn-adi2"));
        #endif

        #ifdef USE_CUDA
        if (cuda_device_visible())
        {
            solver_name_list.push_back("cuda, periodic");
            solver_name_list.push_back("cuda, periodic, reduce_memory");
            solver_list.push_back(make_propagator_computation<double>("cuda", false, new CudaComputationBox<double>({II}, {Lx}, bc_prd), molecules, propagator_computation_optimizer, "cn-adi2"));
            solver_list.push_back(make_propagator_computation<double>("cuda", true, new CudaComputationBox<double>({II}, {Lx}, bc_prd), molecules, propagator_computation_optimizer, "cn-adi2"));
        }
        #endif

        // Reference values of q_next in the boxes of bc_abs, bc_rfl and bc_prd: the output of the CPU
        // solver, which CUDA reproduces. Every solver is checked against them, so that a build without
        // a GPU, whose CPU platforms all run the same CN-ADI code, tests that code as well.
        const double q_reference[3][M] = {
            {  // bc_abs
                0.19215822438755298, 0.42529872610875419, 0.51899217993989577, 0.54797070811135029,
                0.56488876369384466, 0.64115585119885554, 0.64003616849795464, 0.5904772964327355,
                0.4337228349882441, 0.18182822696074835
            },
            {  // bc_rfl
                0.63851503176712354, 0.58188266679575795, 0.56426781341032095, 0.55864553793396954,
                0.56734679504386409, 0.64383783154158647, 0.65114307437193886, 0.63815201321034876,
                0.59945079765492781, 0.63091039807409643
            },
            {  // bc_prd
                0.64589936438207218, 0.58605835855654287, 0.565505702777511, 0.55892136586501062,
                0.56738874377022541, 0.64379235587493189, 0.65086866398197629, 0.63690507726697365,
                0.59521262773521755, 0.62370861621134976
            },
        };

        // For each platform
        for(size_t n=0; n<solver_list.size(); n++)
        {
            PropagatorComputation<double>* solver = solver_list[n];

            for(int i=0; i<M; i++)
                q_next[i] = 0.0;

            //---------------- run --------------------
            std::cout<< std::endl << "Running Pseudo: " << n << ", " << solver_name_list[n] << std::endl;

            Polymer& pc = molecules->get_polymer(0);
            solver->compute_propagators({{"A",w_a},{"B",w_b}},{});
            solver->compute_concentrations();

            solver->get_chain_propagator(q_next, 0, 1, 2, pc.get_block(1,2).n_segment);

            // Against the reference values of the boundary conditions
            const double* q_ref = q_reference[n / repeat];
            for(int i=0; i<M; i++)
                diff_sq[i] = pow(q_ref[i] - q_next[i],2);
            error = sqrt(max_or_nan(diff_sq));
            std::cout<< "Error against the reference values: "<< error << std::endl;
            if (!std::isfinite(error) || error > 1e-7)
                return -1;
        }
        // Release the computations, then their boxes and the shared objects
        for (PropagatorComputation<double>* solver : solver_list)
        {
            ComputationBox<double>* cb = solver->get_cb();
            delete solver;
            delete cb;
        }
        delete propagator_computation_optimizer;
        delete molecules;
        return 0;
    }
    catch(std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
