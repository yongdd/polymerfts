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
        const int KK{3};
        const int M{II*JJ*KK};
        const int NN{4};

        double q_next[M];

        std::array<double,M> diff_sq;
        double error;
        double Lx, Ly, Lz, f;

        f = 0.5;
        Lx = 4.0;
        Ly = 3.0;
        Lz = 2.0;

        // initialize pseudo spectral parameters
        double w_a[M] = {0.183471406e+0,0.623968915e+0,0.731257661e+0,0.997228140e+0,0.961913696e+0,
                        0.792673860e-1,0.429684069e+0,0.290531312e+0,0.453270921e+0,0.199228629e+0,
                        0.754931905e-1,0.226924328e+0,0.936407886e+0,0.979392715e+0,0.464957186e+0,
                        0.742653949e+0,0.368019859e+0,0.885231224e+0,0.406191773e+0,0.653096157e+0,
                        0.567929080e-1,0.568028857e+0,0.144986181e+0,0.466158777e+0,0.573327733e+0,
                        0.136324723e+0,0.819010407e+0,0.271218167e+0,0.626224101e+0,0.398109186e-1,
                        0.860031651e+0,0.338153865e+0,0.688078522e+0,0.564682952e+0,0.222924187e+0,
                        0.306816449e+0,0.316316038e+0,0.640568415e+0,0.702342408e+0,0.632135481e+0,
                        0.649402777e+0,0.647100865e+0,0.370402133e+0,0.691313864e+0,0.447870566e+0,
                        0.757298851e+0,0.586173682e+0,0.766745717e-1,0.504185402e+0,0.812016428e+0,
                        0.217988206e+0,0.273487202e+0,0.937672578e+0,0.570540523e+0,0.409071185e+0,
                        0.391548274e-1,0.663478965e+0,0.260755447e+0,0.503943226e+0,0.979481790e+0
                        };

        double w_b[M] = {0.113822903e-1,0.330673934e+0,0.270138412e+0,0.669606774e+0,0.885344778e-1,
                        0.604752856e+0,0.890062293e+0,0.328557615e+0,0.965824739e+0,0.865399960e+0,
                        0.698893686e+0,0.857947305e+0,0.594897904e+0,0.248187208e+0,0.155686710e+0,
                        0.116803898e+0,0.711146609e+0,0.107610460e+0,0.143034307e+0,0.123131521e+0,
                        0.230387237e+0,0.516274641e+0,0.562366089e-1,0.491449746e+0,0.746656140e+0,
                        0.296108614e+0,0.424987667e+0,0.651538750e+0,0.116745920e+0,0.567790110e+0,
                        0.954487190e+0,0.802476927e-1,0.440223916e+0,0.843025420e+0,0.612864528e+0,
                        0.571893767e+0,0.759625605e+0,0.872255004e+0,0.935065364e+0,0.635565347e+0,
                        0.373711972e-2,0.860683468e+0,0.186492706e+0,0.267880995e+0,0.579305501e+0,
                        0.693549226e+0,0.613843845e+0,0.259811620e-1,0.848915465e+0,0.766111508e+0,
                        0.872008750e+0,0.116289041e+0,0.917713893e+0,0.710076955e+0,0.442712526e+0,
                        0.516722213e+0,0.253395805e+0,0.472950065e-1,0.152934959e+0,0.292486174e+0
                        };

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
            "periodic", "periodic",
        };
        std::vector<std::string> bc_rfl = 
        {
            "reflecting", "reflecting",
            "periodic", "periodic",
            "absorbing", "absorbing",
        };
        std::vector<std::string> bc_prd = 
        {
            "periodic", "periodic",
            "absorbing", "absorbing",
            "reflecting", "reflecting",
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
        solver_list.push_back(make_propagator_computation<double>("cpu-fftw", false, new ComputationBox<double>({II,JJ,KK}, {Lx,Ly,Lz}, bc_abs), molecules, propagator_computation_optimizer, "cn-adi2"));
        solver_list.push_back(make_propagator_computation<double>("cpu-fftw", true, new ComputationBox<double>({II,JJ,KK}, {Lx,Ly,Lz}, bc_abs), molecules, propagator_computation_optimizer, "cn-adi2"));
        #endif

        #ifdef USE_CPU_MKL
        repeat += 2;
        solver_name_list.push_back("cpu-mkl, absorbing");
        solver_name_list.push_back("cpu-mkl, absorbing, reduce_memory");
        solver_list.push_back(make_propagator_computation<double>("cpu-mkl", false, new ComputationBox<double>({II,JJ,KK}, {Lx,Ly,Lz}, bc_abs), molecules, propagator_computation_optimizer, "cn-adi2"));
        solver_list.push_back(make_propagator_computation<double>("cpu-mkl", true, new ComputationBox<double>({II,JJ,KK}, {Lx,Ly,Lz}, bc_abs), molecules, propagator_computation_optimizer, "cn-adi2"));
        #endif

        #ifdef USE_CUDA
        if (cuda_device_visible())
        {
            repeat += 2;
            solver_name_list.push_back("cuda, absorbing");
            solver_name_list.push_back("cuda, absorbing, reduce_memory");
            solver_list.push_back(make_propagator_computation<double>("cuda", false, new CudaComputationBox<double>({II,JJ,KK}, {Lx,Ly,Lz}, bc_abs), molecules, propagator_computation_optimizer, "cn-adi2"));
            solver_list.push_back(make_propagator_computation<double>("cuda", true, new CudaComputationBox<double>({II,JJ,KK}, {Lx,Ly,Lz}, bc_abs), molecules, propagator_computation_optimizer, "cn-adi2"));
        }
        #endif

        #ifdef USE_CPU_FFTW
        solver_name_list.push_back("cpu-fftw, reflecting");
        solver_name_list.push_back("cpu-fftw, reflecting, reduce_memory");
        solver_list.push_back(make_propagator_computation<double>("cpu-fftw", false, new ComputationBox<double>({II,JJ,KK}, {Lx,Ly,Lz}, bc_rfl), molecules, propagator_computation_optimizer, "cn-adi2"));
        solver_list.push_back(make_propagator_computation<double>("cpu-fftw", true, new ComputationBox<double>({II,JJ,KK}, {Lx,Ly,Lz}, bc_rfl), molecules, propagator_computation_optimizer, "cn-adi2"));
        #endif

        #ifdef USE_CPU_MKL
        solver_name_list.push_back("cpu-mkl, reflecting");
        solver_name_list.push_back("cpu-mkl, reflecting, reduce_memory");
        solver_list.push_back(make_propagator_computation<double>("cpu-mkl", false, new ComputationBox<double>({II,JJ,KK}, {Lx,Ly,Lz}, bc_rfl), molecules, propagator_computation_optimizer, "cn-adi2"));
        solver_list.push_back(make_propagator_computation<double>("cpu-mkl", true, new ComputationBox<double>({II,JJ,KK}, {Lx,Ly,Lz}, bc_rfl), molecules, propagator_computation_optimizer, "cn-adi2"));
        #endif

        #ifdef USE_CUDA
        if (cuda_device_visible())
        {
            solver_name_list.push_back("cuda, reflecting");
            solver_name_list.push_back("cuda, reflecting, reduce_memory");
            solver_list.push_back(make_propagator_computation<double>("cuda", false, new CudaComputationBox<double>({II,JJ,KK}, {Lx,Ly,Lz}, bc_rfl), molecules, propagator_computation_optimizer, "cn-adi2"));
            solver_list.push_back(make_propagator_computation<double>("cuda", true, new CudaComputationBox<double>({II,JJ,KK}, {Lx,Ly,Lz}, bc_rfl), molecules, propagator_computation_optimizer, "cn-adi2"));
        }
        #endif

        #ifdef USE_CPU_FFTW
        solver_name_list.push_back("cpu-fftw, periodic");
        solver_name_list.push_back("cpu-fftw, periodic, reduce_memory");
        solver_list.push_back(make_propagator_computation<double>("cpu-fftw", false, new ComputationBox<double>({II,JJ,KK}, {Lx,Ly,Lz}, bc_prd), molecules, propagator_computation_optimizer, "cn-adi2"));
        solver_list.push_back(make_propagator_computation<double>("cpu-fftw", true, new ComputationBox<double>({II,JJ,KK}, {Lx,Ly,Lz}, bc_prd), molecules, propagator_computation_optimizer, "cn-adi2"));
        #endif

        #ifdef USE_CPU_MKL
        solver_name_list.push_back("cpu-mkl, periodic");
        solver_name_list.push_back("cpu-mkl, periodic, reduce_memory");
        solver_list.push_back(make_propagator_computation<double>("cpu-mkl", false, new ComputationBox<double>({II,JJ,KK}, {Lx,Ly,Lz}, bc_prd), molecules, propagator_computation_optimizer, "cn-adi2"));
        solver_list.push_back(make_propagator_computation<double>("cpu-mkl", true, new ComputationBox<double>({II,JJ,KK}, {Lx,Ly,Lz}, bc_prd), molecules, propagator_computation_optimizer, "cn-adi2"));
        #endif

        #ifdef USE_CUDA
        if (cuda_device_visible())
        {
            solver_name_list.push_back("cuda, periodic");
            solver_name_list.push_back("cuda, periodic, reduce_memory");
            solver_list.push_back(make_propagator_computation<double>("cuda", false, new CudaComputationBox<double>({II,JJ,KK}, {Lx,Ly,Lz}, bc_prd), molecules, propagator_computation_optimizer, "cn-adi2"));
            solver_list.push_back(make_propagator_computation<double>("cuda", true, new CudaComputationBox<double>({II,JJ,KK}, {Lx,Ly,Lz}, bc_prd), molecules, propagator_computation_optimizer, "cn-adi2"));
        }
        #endif

        // Reference values of q_next in the boxes of bc_abs, bc_rfl and bc_prd: the output of the CPU
        // solver, which CUDA reproduces. Every solver is checked against them, so that a build without
        // a GPU, whose CPU platforms all run the same CN-ADI code, tests that code as well.
        const double q_reference[3][M] = {
            {  // bc_abs
                0.47045739171582879, 0.40101703561380836, 0.41453939913682064, 0.34549121699345325,
                0.41223817182835709, 0.38504789518633392, 0.34672213482249392, 0.4286174747041534,
                0.34785789219168695, 0.36803624858888245, 0.41277712650360976, 0.3702626782978109,
                0.51940355957611439, 0.5798954877022624, 0.63008289200403966, 0.627738538867185,
                0.55144334083050606, 0.64857915061893823, 0.67780498928851629, 0.69700860154767164,
                0.69803520699725019, 0.60179036664477559, 0.75626964516111694, 0.62744342462831959,
                0.54314147803983626, 0.66863354250044638, 0.59292320759087824, 0.58826377789130235,
                0.70121054206629352, 0.62578208795533585, 0.51253767886779433, 0.7143518967113881,
                0.62012538828992547, 0.52993205486748163, 0.62605306942662142, 0.62390468415559042,
                0.51379488906086757, 0.48500032981213564, 0.46097649388276646, 0.54613007492818866,
                0.64761786383670361, 0.49737103458278076, 0.64504082467969803, 0.62998002192808511,
                0.58252378774997615, 0.54364336504016464, 0.57090359880131791, 0.73652022036848741,
                0.3328719933692893, 0.31404445028217731, 0.33158369708148866, 0.43916213101315099,
                0.3180364652898679, 0.34643018125111752, 0.42108719845240605, 0.41310700170574605,
                0.41544114068269833, 0.47620363431662621, 0.44797280658444288, 0.40981159988951049
            },
            {  // bc_rfl
                0.40066709842765436, 0.54353393632066516, 0.34384856832759564, 0.28063364894417264,
                0.54986321783795722, 0.33156252028386479, 0.28996077703835882, 0.57476711702908656,
                0.29332122271865491, 0.31311037989297408, 0.54006886346558736, 0.30829891674175108,
                0.29307896449786452, 0.53450775650874427, 0.36201267816954646, 0.34478569731344544,
                0.48643652640954443, 0.36327267041326, 0.36975514504450463, 0.61196623544912432,
                0.39025083673688388, 0.32928805644853554, 0.65385143116243449, 0.35236504404804031,
                0.29427322812071216, 0.5779402658803231, 0.32856361732694522, 0.31330986329529353,
                0.59838897349404074, 0.34005847343731305, 0.27380232360873358, 0.6168499364205815,
                0.34200841189906112, 0.28075429124316403, 0.53694659241138698, 0.33801175849735426,
                0.28811010306066626, 0.43278266900570478, 0.26520917145121575, 0.30930234736532441,
                0.57366753741572762, 0.2770602394314573, 0.36104324405167587, 0.55652486204837492,
                0.32151425885813745, 0.29318867438193008, 0.49404296263032244, 0.40108401238866587,
                0.28418593023443761, 0.42424455665606059, 0.2765366143340835, 0.37312105665815876,
                0.41155712329335831, 0.27837957121712964, 0.35908085400868067, 0.55305728795964249,
                0.34526941367935876, 0.40255450770586221, 0.58353313557290587, 0.33095389733118202
            },
            {  // bc_prd
                0.44423060688915234, 0.37519282659729836, 0.38515306908367691, 0.500708497311672,
                0.60392335314895473, 0.58105952441320141, 0.51265409103849524, 0.6335701805036833,
                0.51360338283164397, 0.351400672757749, 0.38604933720224027, 0.34339180687786547,
                0.3232791280822534, 0.36661179782828102, 0.40756286343278347, 0.61334857096469086,
                0.5431736374065802, 0.64504493315484623, 0.65553551753441597, 0.68291835950287028,
                0.69649271318909622, 0.37564356460714327, 0.47209965904635148, 0.40024485334202203,
                0.32989427670265714, 0.40500965800912886, 0.36297413955967184, 0.55144730390403773,
                0.66372757838009122, 0.6029812283088416, 0.47290616636105204, 0.67693451745890454,
                0.60010666261128054, 0.31058287868525264, 0.37980313181960912, 0.38744021065152057,
                0.33069016830769199, 0.30747936746579424, 0.28582144242333885, 0.54506123246008797,
                0.63748829814440422, 0.48558394611694167, 0.64284858721400562, 0.62212499247129194,
                0.56562822238785704, 0.33204512735221786, 0.36191264065993528, 0.47613261089571851,
                0.32540751390037864, 0.30233311945282881, 0.3173182827033395, 0.65856649086346852,
                0.47355381876339919, 0.50685028661363241, 0.62464208199575699, 0.61390185187673296,
                0.60698795533578354, 0.46529709281178266, 0.4307876323375206, 0.37822783495771878
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
