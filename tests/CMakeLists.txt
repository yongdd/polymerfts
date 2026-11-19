# Tests that read input files (Hexagonal*) run with the source directory as
# working directory; the Python tests locate tests/data from their own path.
# Nothing is copied into the build tree.

# Every test carries a label; configuration stops for a test without one:
#   basic      a quick check of every part (interfaces, transforms, solvers, SCFT,
#              stress), which the Windows CI job and the MKL-only Linux CI job run
#              (ctest -L basic -j 4)
#   numerical  the physical and numerical regressions, for a change of a numerical
#              algorithm (ctest -L numerical); some of them are basic too
#   examples   the example scripts
# Only a run without -L covers the whole suite.
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
    TestPropagatorParserSafety
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
    TestCpuFFT1D
    TestCpuFFT1DComplex
    TestCpuFFT2D
    TestCpuFFT2DComplex
    TestCpuFFT3D
    TestCpuFFT3DComplex
    TestCpuFFTAddressIndependence
    TestCpuMakhoulTransform
    TestSimulationBox
    TestCpuTridiagonal
    # Fourier multipliers of whole fields on every platform (Python)
    TestFourierTransform

    # CUDA FFT & utilities
    TestCudaFFT
    TestCudaFFTMixedBC
    TestCudaMakhoulTransform
    TestCudaDCT
    TestCudaDST
    TestCudaSimulationBoxComplex
    TestCudaRealSpaceRecovery
    TestCudaTridiagonal
    TestCudaPinnedMemory
    TestCudaCubReduceSum
    TestCudaCubReduceSumClass
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
    TestCrankNicolson1D
    TestCrankNicolson2D
    TestCrankNicolson3D
    TestDCT2Propagator
    TestDST2Propagator
    TestPseudoLinearContinuous3D
    TestPseudoLinearDiscrete3D
    TestRealSpaceLinearContinuous3D
    TestRealSpaceAbsorbingRange
    TestMultipleLocalDs
    TestInitialConditionPropagator

    # Aggregation tests
    TestAggregation3D
    TestAggregationDendritic
    TestAggregationDiscrete
    TestAggregationSideChains
    TestAggregationSlicedSideChains

    # Branched polymer tests
    TestPseudoBranchedContinuous3D
    TestPseudoBranchedDiscrete3D
    TestPseudoBranchedMixtureContinuous3D
    TestPseudoBranchedMixtureDiscrete3D

    # SCFT & Stress
    TestScft1D
    TestScft2D
    TestStressFiniteDifference

    # Space Group
    TestSpaceGroupReducedBasis
    TestSpaceGroupBasisSelection
    TestSpaceGroupCell
    TestSpaceGroupReduceMemorySafety
    TestMklRealTransform
    TestMklFftwAccuracyAllTypes
)

# Physical and numerical regressions (label "numerical")
SET(NUMERICAL_TESTS
    TestParameterNames TestFtsReliability TestCheckpointValidation TestExtensions TestPropagatorAnalytic TestConvergenceOrderLite TestSamplingStatistics
    TestStructureFunctionRPA TestChargedStructureFunctionRPA TestStructureFunctionRecords TestClftsElectrostatics TestClftsPointIons TestClftsGrandCanonical TestScftElectrostatics TestGibbsEnsemble TestConfinedElectrostatics TestClftsEtd TestClftsThermodynamics
    TestLftsRun TestLftsDetails TestLftsIdealLimit TestSpaceGroupPlatformEquivalence TestObliqueZStress TestScftObliqueSpaceGroups
    TestStressNonPeriodic TestStressNonPeriodicRK2 TestStressComplexWalls TestStressAliasTies
    TestJunctionRange TestBlockMultiplicity
    TestScftAbsentPolymer TestRealSpaceAbsorbingRange TestRealSpaceDiscrete
    TestCnAdiLongLines TestCrysFFTCellChange TestCpuCrysFFTFullGrid TestBoxVolumeElement
    TestContourPrecision TestDiscreteComposition TestCanonicalIonRange TestThinObliqueCell TestFourierTransform TestGceZeroPartition TestGceIonRange TestPhysicalBasisOperators TestSmearingAliases TestWtmdZeroModes
    TestPartitionIntegrand
    # Stress, SCFT and propagators
    TestStressFiniteDifference TestStressSpaceGroup TestStressComplexPlatforms
    TestScft3D TestScftCompressible TestScftHexagonalCylinder2D TestScftOptions TestSpaceGroupPhases TestCrysFFTM3SmallGrid
    TestSmearingLattice TestPhaseEquilibrium TestPolymerFieldTheory TestBlockConcentration TestMinimalFormAlias
    TestPseudoLinearContinuous3DComplex TestPseudoLinearDiscrete3DComplex TestPseudoSolventContinuous3D
    TestPseudoSolventContinuous3DComplex TestPseudoSolventDiscrete3D TestPseudoSolventDiscrete3DComplex
    TestNanoParticle2D TestNonMonotonicConvergence TestHexagonalRectangularEquivalence
    # Boundary conditions and platforms
    TestAbsorbingVsMask TestComplexNonPeriodicMirror TestCompareRealSpacePseudoSpectral
    TestCpuCudaNonPeriodicEquivalence TestCudaOddReflecting
)

SET(EXAMPLES_TESTS
    TestExamples
)

# Every name in the lists is a test of this directory
FOREACH(TEST_NAME IN LISTS BASIC_TESTS NUMERICAL_TESTS EXAMPLES_TESTS)
    IF(NOT TEST_NAME STREQUAL "TestCudaDeviceBufferOutOfMemory"
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

# Windows looks for the DLLs of an executable on PATH. The spglib DLL lies in
# the spglib Python package, above the directory of its import library.
FUNCTION(SET_TEST_DLL_PATH TEST_NAME)
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
# not baked in.
FILE(GLOB FILE_LIST "Test*.py")
FOREACH(FILE_NAME ${FILE_LIST})
    GET_FILENAME_COMPONENT(TEST_NAME ${FILE_NAME} NAME_WE)
    ADD_TEST(NAME ${TEST_NAME} COMMAND ${Python3_EXECUTABLE} ${FILE_NAME})
    # Exit code 77: skipped (e.g. the test needs a CPU platform this build lacks)
    # Python writes no tests/__pycache__ into the source tree
    SET_TESTS_PROPERTIES(${TEST_NAME} PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python;PYTHONDONTWRITEBYTECODE=1"
        SKIP_RETURN_CODE 77)
    SET_TEST_LABELS(${TEST_NAME})
ENDFOREACH()

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

    IF(${TEST_NAME} MATCHES "Hexagonal")
        ADD_TEST(NAME ${TEST_NAME}
            COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            )
    ELSE()
        ADD_TEST(${TEST_NAME} ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME})
    ENDIF()
    # Exit code 77: skipped (a CPU platform the test needs is missing, e.g. cpu-fftw
    # in an MKL-only build). A CUDA build run without a visible GPU fails instead.
    SET_TESTS_PROPERTIES(${TEST_NAME} PROPERTIES SKIP_RETURN_CODE 77)
    SET_TEST_LABELS(${TEST_NAME})
    SET_TEST_DLL_PATH(${TEST_NAME})
ENDFOREACH()

# Benchmark executables (not added to ctest)
FILE(GLOB BENCHMARK_LIST "Benchmark*.cpp")
FOREACH(FILE_NAME ${BENCHMARK_LIST})
    GET_FILENAME_COMPONENT(BENCH_NAME ${FILE_NAME} NAME_WE)
    # Skip platform-specific benchmarks if platform is not available
    IF((${BENCH_NAME} MATCHES "Cuda") AND NOT POLYMERFTS_BUILD_CUDA)
        CONTINUE()
    ELSEIF((${BENCH_NAME} MATCHES "Mkl") AND NOT BUILD_CPU_MKL_LIB)
        CONTINUE()
    ELSEIF((${BENCH_NAME} MATCHES "Fftw") AND NOT BUILD_CPU_FFTW_LIB)
        CONTINUE()
    ENDIF()

    ADD_EXECUTABLE(${BENCH_NAME} ${FILE_NAME})
    TARGET_LINK_LIBRARIES(${BENCH_NAME} PRIVATE polymerfts_backends)
ENDFOREACH()

IF(POLYMERFTS_BUILD_CUDA)
    # Generate files named "Test*.cu"
    FILE(GLOB FILE_LIST "Test*.cu")
    FOREACH(FILE_NAME ${FILE_LIST})
        GET_FILENAME_COMPONENT(TEST_NAME ${FILE_NAME} NAME_WE)

        # Several CUDA tests use FFTW as the CPU reference (#include <fftw3.h>
        # without a USE_CPU_FFTW guard); they cannot build without it.
        IF((${TEST_NAME} MATCHES "Fftw|CrysFFT|CudaDCT|CudaDST") AND NOT BUILD_CPU_FFTW_LIB)
            CONTINUE()
        ENDIF()

        ADD_EXECUTABLE(${TEST_NAME} ${FILE_NAME})
        TARGET_LINK_LIBRARIES(${TEST_NAME} PRIVATE polymerfts_backends)

        # Same architecture list as the cuda library
        SET_PROPERTY(TARGET ${TEST_NAME} PROPERTY CUDA_ARCHITECTURES ${POLYMERFTS_CUDA_ARCHITECTURES})
        # Benchmarks (Test*Bench*.cu) build as executables but are not tests.
        IF(NOT ${TEST_NAME} MATCHES "Bench")
            ADD_TEST(${TEST_NAME} ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME})
            SET_TEST_LABELS(${TEST_NAME})
            SET_TEST_DLL_PATH(${TEST_NAME})
        ENDIF()
    ENDFOREACH()
    # Its out-of-memory check (--out-of-memory) holds most of the free device memory
    # for a moment (a ballast buffer): a separate entry, kept away from the other
    # tests under ctest -j, so that its skip does not hide the other checks
    IF(TARGET TestCudaDeviceBuffer)
        ADD_TEST(TestCudaDeviceBufferOutOfMemory ${CMAKE_CURRENT_BINARY_DIR}/TestCudaDeviceBuffer --out-of-memory)
        SET_TEST_DLL_PATH(TestCudaDeviceBufferOutOfMemory)
        SET_TEST_LABELS(TestCudaDeviceBufferOutOfMemory)
        # Exit code 77: the device memory could not be held, or the driver backs it with
        # system memory (the WDDM sysmem fallback), so no construction can run out
        SET_TESTS_PROPERTIES(TestCudaDeviceBufferOutOfMemory PROPERTIES RUN_SERIAL TRUE SKIP_RETURN_CODE 77)
        # NVML gives this process's device memory for the end-to-end check
        # (other processes on the GPU then do not disturb it)
        IF(TARGET CUDA::nvml)
            TARGET_LINK_LIBRARIES(TestCudaDeviceBuffer PRIVATE CUDA::nvml)
            TARGET_COMPILE_DEFINITIONS(TestCudaDeviceBuffer PRIVATE POLYMERFTS_TEST_HAS_NVML)
        ENDIF()
    ENDIF()
ENDIF()
