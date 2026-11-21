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
        const int II{5};
        const int JJ{4};
        const int M{II*JJ};
        const int NN{4};

        double q_next[M];

        std::array<double,M> diff_sq;
        double error;
        double Lx, Ly, f;

        f = 0.5;
        Lx = 4.0;
        Ly = 3.0;

        double w_a[M] = {0.822383458999126,  0.180877073118435, 0.885692320279145,
                         0.89417010799111,   0.495074990864166, 0.612975741629382,
                         0.0415795198090432, 0.353431810889399, 0.773118461366249,
                         0.474587294381635,
                         0.200002276650706, 0.592127025743285, 0.460207620078036,
                         0.435945198378862, 0.61269588805607,  0.355979618324841,
                         0.548759176402544, 0.482897565408353, 0.541501788021353,
                         0.349682106604464};

        double w_b[M] = {0.104943438436657, 0.418086769592863, 0.61190542613784,
                         0.792961240687622, 0.713098832553561, 0.667410867822433,
                         0.492427261460169, 0.261956970404376, 0.479635996452285,
                         0.206215439022739,
                         0.180066106322814, 0.349989269840229, 0.580249533529743,
                         0.653060847246207, 0.793729416310673, 0.988032605316576,
                         0.98005550969782,  0.38678227079795,  0.894395839154923,
                         0.720491484453521};

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
            "reflecting", "reflecting",
        };
        std::vector<std::string> bc_rfl = 
        {
            "reflecting", "reflecting",
            "periodic", "periodic",
        };
        std::vector<std::string> bc_prd = 
        {
            "periodic", "periodic",
            "absorbing", "absorbing",
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
        solver_list.push_back(make_propagator_computation<double>("cpu-fftw", false, new ComputationBox<double>({II,JJ}, {Lx,Ly}, bc_abs), molecules, propagator_computation_optimizer, "cn-adi2"));
        solver_list.push_back(make_propagator_computation<double>("cpu-fftw", true, new ComputationBox<double>({II,JJ}, {Lx,Ly}, bc_abs), molecules, propagator_computation_optimizer, "cn-adi2"));
        #endif

        #ifdef USE_CPU_MKL
        repeat += 2;
        solver_name_list.push_back("cpu-mkl, absorbing");
        solver_name_list.push_back("cpu-mkl, absorbing, reduce_memory");
        solver_list.push_back(make_propagator_computation<double>("cpu-mkl", false, new ComputationBox<double>({II,JJ}, {Lx,Ly}, bc_abs), molecules, propagator_computation_optimizer, "cn-adi2"));
        solver_list.push_back(make_propagator_computation<double>("cpu-mkl", true, new ComputationBox<double>({II,JJ}, {Lx,Ly}, bc_abs), molecules, propagator_computation_optimizer, "cn-adi2"));
        #endif

        #ifdef USE_CUDA
        if (cuda_device_visible())
        {
            repeat += 2;
            solver_name_list.push_back("cuda, absorbing");
            solver_name_list.push_back("cuda, absorbing, reduce_memory");
            solver_list.push_back(make_propagator_computation<double>("cuda", false, new CudaComputationBox<double>({II,JJ}, {Lx,Ly}, bc_abs), molecules, propagator_computation_optimizer, "cn-adi2"));
            solver_list.push_back(make_propagator_computation<double>("cuda", true, new CudaComputationBox<double>({II,JJ}, {Lx,Ly}, bc_abs), molecules, propagator_computation_optimizer, "cn-adi2"));
        }
        #endif

        #ifdef USE_CPU_FFTW
        solver_name_list.push_back("cpu-fftw, reflecting");
        solver_name_list.push_back("cpu-fftw, reflecting, reduce_memory");
        solver_list.push_back(make_propagator_computation<double>("cpu-fftw", false, new ComputationBox<double>({II,JJ}, {Lx,Ly}, bc_rfl), molecules, propagator_computation_optimizer, "cn-adi2"));
        solver_list.push_back(make_propagator_computation<double>("cpu-fftw", true, new ComputationBox<double>({II,JJ}, {Lx,Ly}, bc_rfl), molecules, propagator_computation_optimizer, "cn-adi2"));
        #endif

        #ifdef USE_CPU_MKL
        solver_name_list.push_back("cpu-mkl, reflecting");
        solver_name_list.push_back("cpu-mkl, reflecting, reduce_memory");
        solver_list.push_back(make_propagator_computation<double>("cpu-mkl", false, new ComputationBox<double>({II,JJ}, {Lx,Ly}, bc_rfl), molecules, propagator_computation_optimizer, "cn-adi2"));
        solver_list.push_back(make_propagator_computation<double>("cpu-mkl", true, new ComputationBox<double>({II,JJ}, {Lx,Ly}, bc_rfl), molecules, propagator_computation_optimizer, "cn-adi2"));
        #endif

        #ifdef USE_CUDA
        if (cuda_device_visible())
        {
            solver_name_list.push_back("cuda, reflecting");
            solver_name_list.push_back("cuda, reflecting, reduce_memory");
            solver_list.push_back(make_propagator_computation<double>("cuda", false, new CudaComputationBox<double>({II,JJ}, {Lx,Ly}, bc_rfl), molecules, propagator_computation_optimizer, "cn-adi2"));
            solver_list.push_back(make_propagator_computation<double>("cuda", true, new CudaComputationBox<double>({II,JJ}, {Lx,Ly}, bc_rfl), molecules, propagator_computation_optimizer, "cn-adi2"));
        }
        #endif

        #ifdef USE_CPU_FFTW
        solver_name_list.push_back("cpu-fftw, periodic");
        solver_name_list.push_back("cpu-fftw, periodic, reduce_memory");
        solver_list.push_back(make_propagator_computation<double>("cpu-fftw", false, new ComputationBox<double>({II,JJ}, {Lx,Ly}, bc_prd), molecules, propagator_computation_optimizer, "cn-adi2"));
        solver_list.push_back(make_propagator_computation<double>("cpu-fftw", true, new ComputationBox<double>({II,JJ}, {Lx,Ly}, bc_prd), molecules, propagator_computation_optimizer, "cn-adi2"));
        #endif

        #ifdef USE_CPU_MKL
        solver_name_list.push_back("cpu-mkl, periodic");
        solver_name_list.push_back("cpu-mkl, periodic, reduce_memory");
        solver_list.push_back(make_propagator_computation<double>("cpu-mkl", false, new ComputationBox<double>({II,JJ}, {Lx,Ly}, bc_prd), molecules, propagator_computation_optimizer, "cn-adi2"));
        solver_list.push_back(make_propagator_computation<double>("cpu-mkl", true, new ComputationBox<double>({II,JJ}, {Lx,Ly}, bc_prd), molecules, propagator_computation_optimizer, "cn-adi2"));
        #endif

        #ifdef USE_CUDA
        if (cuda_device_visible())
        {
            solver_name_list.push_back("cuda, periodic");
            solver_name_list.push_back("cuda, periodic, reduce_memory");
            solver_list.push_back(make_propagator_computation<double>("cuda", false, new CudaComputationBox<double>({II,JJ}, {Lx,Ly}, bc_prd), molecules, propagator_computation_optimizer, "cn-adi2"));
            solver_list.push_back(make_propagator_computation<double>("cuda", true, new CudaComputationBox<double>({II,JJ}, {Lx,Ly}, bc_prd), molecules, propagator_computation_optimizer, "cn-adi2"));
        }
        #endif

        // Reference values of q_next in the boxes of bc_abs, bc_rfl and bc_prd: the output of the CPU
        // solver, which CUDA reproduces. Every solver is checked against them, so that a build without
        // a GPU, whose CPU platforms all run the same CN-ADI code, tests that code as well.
        const double q_reference[3][M] = {
            {  // bc_abs
                0.41565103137540887, 0.41871070684552214, 0.34271971656680872, 0.302863044003153,
                0.51741205242335331, 0.54980260060528208, 0.66216305043766321, 0.675556570325336,
                0.56471825243850915, 0.68441943772772951, 0.7497787288975114, 0.64969355642163518,
                0.55299585805242979, 0.5525844632622835, 0.50986693465577104, 0.48951043026252195,
                0.30966369106844083, 0.39324968772123436, 0.32309976916367822, 0.35986369750643854
            },
            {  // bc_rfl
                0.64273991065539648, 0.66289270694540514, 0.53204515154681342, 0.47936227651127034,
                0.5551361041774866, 0.5785849532299806, 0.68559285366858591, 0.68537184521911765,
                0.57892402442923296, 0.69032179749600808, 0.75344427032088923, 0.64374660604263179,
                0.57673085386047884, 0.57912094448062201, 0.53292059309890061, 0.51393699134342552,
                0.49464568094727068, 0.62163849604726007, 0.5083058676721397, 0.55920420143228644
            },
            {  // bc_prd
                0.39514024271393278, 0.62155622696980906, 0.50531610984629083, 0.29174766464782026,
                0.32745110075037992, 0.54393270246787817, 0.64789746857729968, 0.42335800188747807,
                0.3458973182892387, 0.65257871489731611, 0.71382085832644049, 0.39728686769662624,
                0.34749576690309769, 0.54507241663954886, 0.5001508588771848, 0.30737843331387305,
                0.29833972535356573, 0.59283867965774806, 0.47301852303973752, 0.33025968052026555
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
