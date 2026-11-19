/**
 * @file TestCudaPart.h
 * @brief Fail a platform-loop test whose CUDA part did not run, and run a part
 *        with a chosen number of streams.
 *
 * PlatformSelector::avail_platforms() lists "cuda" only when a CUDA device is
 * visible. A test that loops over those platforms would, in a build with the
 * CUDA backend on a node without a GPU, run only its CPU parts and pass, which
 * reads as a passed CUDA test. Such a test runs its CPU parts and then ends
 * with `return cuda_part_missing() ? 1 : 0;`, like the TestCuda* tests and the
 * Python tests (tests/cuda_part.py).
 *
 * A propagator computation takes its number of CUDA streams (or CPU threads)
 * from OMP_NUM_THREADS when it is created. ScopedStreams sets it for a scope, so
 * that a test runs several streams whatever the environment of ctest is.
 */

#ifndef TEST_CUDA_PART_H_
#define TEST_CUDA_PART_H_

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "PlatformSelector.h"

/// True, after printing why, if this build has the CUDA backend but no CUDA device is visible
inline bool cuda_part_missing()
{
#ifdef USE_CUDA
    std::vector<std::string> platforms = PlatformSelector::avail_platforms();
    if (std::find(platforms.begin(), platforms.end(), "cuda") == platforms.end())
    {
        std::cout << "FAILED: this build has the CUDA backend, but no CUDA device is visible: "
                  << "the CUDA part of this test did not run." << std::endl;
        return true;
    }
#endif
    return false;
}

/// OMP_NUM_THREADS set to n for a scope, the caller's value restored afterwards. A
/// propagator computation created in the scope runs n CUDA streams (at most 4) or n
/// CPU threads.
struct ScopedStreams
{
    explicit ScopedStreams(const char* n)
    {
        const char* value = std::getenv("OMP_NUM_THREADS");
        had = (value != nullptr);
        if (had)
            saved = value;
        set("OMP_NUM_THREADS", n);
    }
    ~ScopedStreams()
    {
        if (had)
            set("OMP_NUM_THREADS", saved.c_str());
        else
            unset("OMP_NUM_THREADS");
    }
    ScopedStreams(const ScopedStreams&) = delete;
    ScopedStreams& operator=(const ScopedStreams&) = delete;

    bool had = false;
    std::string saved;

private:
#ifdef _WIN32
    // The CRT environment that getenv reads; an empty value removes the variable
    static void set(const char* name, const char* value) { _putenv_s(name, value); }
    static void unset(const char* name) { _putenv_s(name, ""); }
#else
    static void set(const char* name, const char* value) { setenv(name, value, 1); }
    static void unset(const char* name) { unsetenv(name); }
#endif
};

#endif
