# The native tests read their input files from tests/data through TEST_DATA_DIR
# (TestHexagonalRectangularEquivalence its initial fields); the Python tests locate
# tests/data from their own path. Nothing is copied into the build tree.
ADD_COMPILE_DEFINITIONS(TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

# Every test carries a label; configuration stops for a test without one:
#   basic      a quick check of every part (interfaces, transforms, solvers, SCFT,
#              stress), which the Windows CI job runs (ctest -L basic -j 4)
#   numerical  the physical and numerical regressions, for a change of a numerical
#              algorithm (ctest -L numerical); some of them are basic too
#   examples   the example scripts and the tutorial notebooks
# Only a run without -L covers the whole suite, as the Linux CI jobs run it.
SET(BASIC_TESTS
    # Parameter handling of the simulation classes and array handling of the
    # bindings (CPU, a few seconds each)
    TestParams
    TestParameterNames
    TestFtsReliability
    TestCheckpointValidation
    TestCheckpointCompatibility
    TestPairRecords
    TestWtmdLifecycle
    TestAndersonMixing
    TestAbsentPointIons
    TestAbsentSolvent
    TestScftInitialPropagators
    TestScftAbsentPolymer
    TestScftInitialFields
    TestPeriodicBoundaryCase
    TestScftResultGeometry
    TestMaskedDiscreteWeights
    TestSmallFieldCoefficients
    TestJunctionRange
    TestJunctionProduct
    TestCanonicalIonRange
    TestThinObliqueCell
    TestRandomCopolymerFractions
    TestNumpyScalarSerialization
    TestMatFieldNames
    TestGceIonRange
    TestVtkFieldIntegrity
    TestExtensions
    TestPropagatorAnalytic
    TestComplexMask
    TestComputationSafety
    TestComputationState
    TestConcentrationsAndStress
    TestInputChecks
    TestErrorMessages
    TestMoleculesValidation
    TestPartitionGuard
    TestPartitionIntegrand
    TestDriftStepHistory
    TestChains
    TestRunRecord
    TestSimulationRelease
    TestPackageHashes
    # Platform & utilities
    TestConsoleEncoding
    TestOpenMPDefaults
    TestDisplayArchitecture
    TestCircularBuffer
    TestException
    TestScheduler
    TestPropagatorCode
    TestPropagatorCodeStack
    TestPropagatorCodeReduction
    TestCompareBranchKey
    TestContourLengthMapping
    TestContourPrecision
    TestDiscreteComposition
    TestGceZeroPartition
    TestPhysicalBasisOperators
    TestSmearingAliases
    TestWtmdZeroModes
    TestWtmdRange
    TestCompressors
    TestConfigFile
    TestGridLimits
    TestPropagatorKeys
    TestSaveVtk
    TestScftCallback
    TestScftResult
    TestPartitionFunctionCheck

    # CPU FFT (all types)
    TestCpuFFT
    TestCpuFFTAddressIndependence
    TestCpuMakhoulTransform
    TestSimulationBox
    TestCpuTridiagonal
    # Fourier multipliers of whole fields on every platform (Python)
    TestFourierTransform

    # CUDA FFT & utilities
    TestCudaMakhoulTransform
    TestCudaSimulationBoxComplex
    TestCudaRealSpaceRecovery
    TestCudaTridiagonal
    TestCrysFFT
    TestCrysFFTObliqueZGlide
    TestCrysFFTObliqueZCuda
    TestCrysFFTSelector
    TestCudaCrysTablesAllocation
    TestCudaCrysFFTStreams
    TestCudaDeviceBuffer
    TestCudaDeviceBufferOutOfMemory
    TestCpuCrysFFTTables

    # Propagator solvers
    TestCrankNicolson
    TestDCTDSTPropagator
    TestPseudoLinear3D
    TestRealSpaceLinearContinuous3D
    TestRealSpaceAbsorbingRange
    TestMultipleLocalDs
    TestInitialConditionPropagator

    # Aggregation tests
    TestAggregation
    TestAggregationSlicedSideChains

    # Branched polymer tests
    TestPseudoBranched3D
    TestPseudoBranchedMixture3D

    # SCFT & Stress
    TestScft
    TestStressFiniteDifference

    # Space Group
    TestSpaceGroupReducedBasis
    TestSpaceGroupOrbits
    TestSpaceGroupBasisSelection
    TestSpaceGroupCell
    TestSpaceGroupReduceMemorySafety
    TestMklRealTransform
    TestMklFftwAccuracyAllTypes
)

# Physical and numerical regressions (label "numerical")
SET(NUMERICAL_TESTS
    TestParameterNames TestFtsReliability TestCheckpointValidation TestExtensions TestPropagatorAnalytic TestConvergenceOrderLite TestSamplingStatistics
    TestStructureFunctionRPA TestChargedStructureFunctionRPA TestChargedStructureFunctionRPAPolyampholyte TestStructureFunctionRecords TestClftsElectrostatics TestClftsPointIons TestClftsGrandCanonical TestScftElectrostatics TestGibbsEnsemble TestConfinedElectrostatics TestClftsEtd TestClftsThermodynamics
    TestLftsRun TestLftsDetails TestLftsIdealLimit TestSpaceGroupPlatformEquivalence TestObliqueZStress TestScftObliqueSpaceGroups
    TestStressNonPeriodic TestStressComplexWalls TestStressAliasTies
    TestJunctionRange TestBlockMultiplicity
    TestScftAbsentPolymer TestRealSpaceAbsorbingRange TestRealSpaceDiscrete
    TestCnAdiLongLines TestCrysFFTCellChange TestCpuCrysFFTFullGrid TestBoxVolumeElement
    TestContourPrecision TestDiscreteComposition TestCanonicalIonRange TestThinObliqueCell TestFourierTransform TestGceZeroPartition TestGceIonRange TestPhysicalBasisOperators TestSmearingAliases TestWtmdZeroModes
    TestPartitionIntegrand TestConcentrationsAndStress TestCrankNicolson
    # Stress, SCFT and propagators
    TestStressFiniteDifference TestStressSpaceGroup TestStressComplexPlatforms
    TestScft3D TestScftCompressible TestScftHexagonalCylinder2D TestScftOptions TestSpaceGroupPhases TestCrysFFTM3SmallGrid
    TestSmearingLattice TestPhaseEquilibrium TestPolymerFieldTheory TestBlockConcentration TestMinimalFormAlias
    TestPseudoLinear3D TestPseudoSolvent3D
    TestNanoParticle2D TestNonMonotonicConvergence TestHexagonalRectangularEquivalence
    # Boundary conditions and platforms
    TestAbsorbingVsMask TestComplexNonPeriodicMirror TestCompareRealSpacePseudoSpectral
    TestWallPlatformEquivalence
)

SET(EXAMPLES_TESTS
    TestExamples
    TestTutorials
)

# CTest entries that run one case of a test file (an argument of the test) and are named in
# the lists like a test: a long case runs beside the others under ctest -j, or keeps a label
# of its own. Each is added below, after its test.
SET(CASE_ENTRIES
    TestCudaDeviceBufferOutOfMemory                 # TestCudaDeviceBuffer --out-of-memory
    TestScft3D                                      # TestScft --3d
    TestChargedStructureFunctionRPAPolyampholyte    # TestChargedStructureFunctionRPA.py --polyampholyte
)

# Every name in the lists is a test of this directory
FOREACH(TEST_NAME IN LISTS BASIC_TESTS NUMERICAL_TESTS EXAMPLES_TESTS)
    IF(NOT TEST_NAME IN_LIST CASE_ENTRIES
       AND NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_NAME}.py
       AND NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_NAME}.cpp
       AND NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_NAME}.cu)
        MESSAGE(FATAL_ERROR "tests/CMakeLists.txt labels ${TEST_NAME}, which is not a test of tests/.")
    ENDIF()
ENDFOREACH()

# The labels of a test from the lists above
FUNCTION(SET_TEST_LABELS TEST_NAME)
    SET(_labels "")
    IF(${TEST_NAME} IN_LIST BASIC_TESTS)
        LIST(APPEND _labels basic)
    ENDIF()
    IF(${TEST_NAME} IN_LIST NUMERICAL_TESTS)
        LIST(APPEND _labels numerical)
    ENDIF()
    IF(${TEST_NAME} IN_LIST EXAMPLES_TESTS)
        LIST(APPEND _labels examples)
    ENDIF()
    IF(NOT _labels)
        MESSAGE(FATAL_ERROR "${TEST_NAME} has no label: add it to BASIC_TESTS, NUMERICAL_TESTS or "
            "EXAMPLES_TESTS in tests/CMakeLists.txt.")
    ENDIF()
    SET_TESTS_PROPERTIES(${TEST_NAME} PROPERTIES LABELS "${_labels}")
ENDFUNCTION()

# A CTest entry: the command (an executable target stands for its path), its labels, exit
# code 77 for a skip (e.g. the test needs a CPU platform this build lacks; a CUDA build run
# without a visible GPU fails instead) and on Windows the DLL path: Windows looks for the
# DLLs of an executable on PATH, and the spglib DLL lies in the spglib Python package,
# above the directory of its import library.
FUNCTION(POLYMERFTS_ADD_TEST TEST_NAME)
    ADD_TEST(NAME ${TEST_NAME} COMMAND ${ARGN})
    SET_TESTS_PROPERTIES(${TEST_NAME} PROPERTIES SKIP_RETURN_CODE 77)
    SET_TEST_LABELS(${TEST_NAME})
    IF(WIN32)
        GET_FILENAME_COMPONENT(_spglib_lib_dir "${SPGLIB_LIBRARIES}" DIRECTORY)
        GET_FILENAME_COMPONENT(_spglib_root "${_spglib_lib_dir}" DIRECTORY)
        SET_PROPERTY(TEST ${TEST_NAME} PROPERTY ENVIRONMENT_MODIFICATION
            "PATH=path_list_prepend:${_spglib_root}" "PATH=path_list_prepend:${_spglib_root}/bin")
    ENDIF()
ENDFUNCTION()

# Add Python Files. They import the package staged in the build tree
# (see polymerfts_stage in the top-level CMakeLists.txt), so installation
# is not required and an old site-packages install is never tested by mistake.
# Only the staged package is on PYTHONPATH; the configuring shell's value is
# not baked in. Python writes no tests/__pycache__ into the source tree.
FILE(GLOB FILE_LIST "Test*.py")
FOREACH(FILE_NAME ${FILE_LIST})
    GET_FILENAME_COMPONENT(TEST_NAME ${FILE_NAME} NAME_WE)
    LIST(APPEND PYTHON_TESTS ${TEST_NAME})
    POLYMERFTS_ADD_TEST(${TEST_NAME} ${Python3_EXECUTABLE} ${FILE_NAME})
ENDFOREACH()
# The polyampholyte of TestChargedStructureFunctionRPA, which runs the diblock
POLYMERFTS_ADD_TEST(TestChargedStructureFunctionRPAPolyampholyte
    ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/TestChargedStructureFunctionRPA.py --polyampholyte)
SET_TESTS_PROPERTIES(${PYTHON_TESTS} TestChargedStructureFunctionRPAPolyampholyte PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python;PYTHONDONTWRITEBYTECODE=1")

# Generate files named "Test*.cpp"
FILE(GLOB FILE_LIST "Test*.cpp")
FOREACH(FILE_NAME ${FILE_LIST})
    GET_FILENAME_COMPONENT(TEST_NAME ${FILE_NAME} NAME_WE)
    IF ((${TEST_NAME} MATCHES "Cuda") AND NOT POLYMERFTS_BUILD_CUDA)
        CONTINUE()
    # "Cpu" tests are backend-agnostic (sources select FFTW/MKL via #ifdef),
    # so they only need SOME CPU backend, not FFTW specifically.
    ELSEIF((${TEST_NAME} MATCHES "Cpu") AND NOT BUILD_CPU_FFTW_LIB AND NOT BUILD_CPU_MKL_LIB)
        CONTINUE()
    ELSEIF((${TEST_NAME} MATCHES "Fftw") AND NOT BUILD_CPU_FFTW_LIB)
        CONTINUE()
    ELSEIF((${TEST_NAME} MATCHES "Mkl") AND NOT BUILD_CPU_MKL_LIB)
        CONTINUE()
    ENDIF()

    ADD_EXECUTABLE(${TEST_NAME} ${FILE_NAME})
    TARGET_LINK_LIBRARIES(${TEST_NAME} PRIVATE polymerfts_backends)
    POLYMERFTS_ADD_TEST(${TEST_NAME} ${TEST_NAME})
ENDFOREACH()
# The 3D box of TestScft (numerical; TestScft runs the 1D and 2D boxes)
POLYMERFTS_ADD_TEST(TestScft3D TestScft --3d)

# Benchmark executables (not added to ctest); Benchmark*.cu needs the CUDA build
FILE(GLOB BENCHMARK_LIST "Benchmark*.cpp")
IF(POLYMERFTS_BUILD_CUDA)
    FILE(GLOB CUDA_BENCHMARK_LIST "Benchmark*.cu")
    LIST(APPEND BENCHMARK_LIST ${CUDA_BENCHMARK_LIST})
ENDIF()
FOREACH(FILE_NAME ${BENCHMARK_LIST})
    GET_FILENAME_COMPONENT(BENCH_NAME ${FILE_NAME} NAME_WE)
    # Skip platform-specific benchmarks if platform is not available; the CrysFFT
    # benchmark compares with FFTW
    IF((${BENCH_NAME} MATCHES "Cuda") AND NOT POLYMERFTS_BUILD_CUDA)
        CONTINUE()
    ELSEIF((${BENCH_NAME} MATCHES "Mkl") AND NOT BUILD_CPU_MKL_LIB)
        CONTINUE()
    ELSEIF((${BENCH_NAME} MATCHES "Fftw|CrysFFT") AND NOT BUILD_CPU_FFTW_LIB)
        CONTINUE()
    ENDIF()

    ADD_EXECUTABLE(${BENCH_NAME} ${FILE_NAME})
    TARGET_LINK_LIBRARIES(${BENCH_NAME} PRIVATE polymerfts_backends)
    IF(FILE_NAME MATCHES "\\.cu$")
        SET_PROPERTY(TARGET ${BENCH_NAME} PROPERTY CUDA_ARCHITECTURES ${POLYMERFTS_CUDA_ARCHITECTURES})
    ENDIF()
ENDFOREACH()

IF(POLYMERFTS_BUILD_CUDA)
    # Generate files named "Test*.cu": the tests with device code of their own (a test
    # that only calls the library is a .cpp and needs "Cuda" in its name)
    FILE(GLOB FILE_LIST "Test*.cu")
    FOREACH(FILE_NAME ${FILE_LIST})
        GET_FILENAME_COMPONENT(TEST_NAME ${FILE_NAME} NAME_WE)

        # The CrysFFT tests use FFTW as the CPU reference (#include <fftw3.h>
        # without a USE_CPU_FFTW guard); they cannot build without it.
        IF((${TEST_NAME} MATCHES "Fftw|CrysFFT") AND NOT BUILD_CPU_FFTW_LIB)
            CONTINUE()
        ENDIF()

        ADD_EXECUTABLE(${TEST_NAME} ${FILE_NAME})
        TARGET_LINK_LIBRARIES(${TEST_NAME} PRIVATE polymerfts_backends)
        # Same architecture list as the cuda library
        SET_PROPERTY(TARGET ${TEST_NAME} PROPERTY CUDA_ARCHITECTURES ${POLYMERFTS_CUDA_ARCHITECTURES})
        POLYMERFTS_ADD_TEST(${TEST_NAME} ${TEST_NAME})
    ENDFOREACH()
    # Its out-of-memory check (--out-of-memory) holds most of the free device memory
    # for a moment (a ballast buffer): a separate entry, kept away from the other
    # tests under ctest -j, so that its skip does not hide the other checks. Exit code
    # 77: the device memory could not be held, or the driver backs it with system
    # memory (the WDDM sysmem fallback), so no construction can run out
    IF(TARGET TestCudaDeviceBuffer)
        POLYMERFTS_ADD_TEST(TestCudaDeviceBufferOutOfMemory TestCudaDeviceBuffer --out-of-memory)
        SET_TESTS_PROPERTIES(TestCudaDeviceBufferOutOfMemory PROPERTIES RUN_SERIAL TRUE)
        # NVML gives this process's device memory for the end-to-end check
        # (other processes on the GPU then do not disturb it)
        IF(TARGET CUDA::nvml)
            TARGET_LINK_LIBRARIES(TestCudaDeviceBuffer PRIVATE CUDA::nvml)
            TARGET_COMPILE_DEFINITIONS(TestCudaDeviceBuffer PRIVATE POLYMERFTS_TEST_HAS_NVML)
        ENDIF()
    ENDIF()
ENDIF()
