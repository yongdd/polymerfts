/**
 * @file TestChecks.h
 * @brief Counted checks of a table-driven test.
 *
 * A table-driven test runs every case on every platform and memory mode, reports
 * every failed check rather than stopping at the first, and ends by printing and
 * asserting the number of runs it executed. A check fails on NaN.
 */

#ifndef TEST_CHECKS_H_
#define TEST_CHECKS_H_

#include <cmath>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "TestNanSafe.h"

/// The checks and runs of a test
struct Checks
{
    int n_checks = 0;
    int n_failed = 0;
    int n_runs = 0;
    int n_failed_runs = 0;

    /// A run: a case on one platform and memory mode
    void begin_run(const std::string& name)
    {
        run_name = name;
        failed_before_run = n_failed;
        std::cout << std::endl << "Running: " << name << std::endl;
    }
    void end_run()
    {
        ++n_runs;
        if (n_failed > failed_before_run)
        {
            ++n_failed_runs;
            std::cout << "FAILED: " << run_name << std::endl;
        }
    }

    /// error <= tolerance
    bool close(const std::string& what, double error, double tolerance)
    {
        const bool ok = std::isfinite(error) && error <= tolerance;
        std::cout << "  " << what << " error: " << error;
        if (!ok)
            std::cout << "  FAILED (tolerance " << tolerance << ")";
        std::cout << std::endl;
        return count(ok);
    }

    /// A condition without an error value
    bool that(const std::string& what, bool ok)
    {
        std::cout << "  " << what << ": " << (ok ? "ok" : "FAILED") << std::endl;
        return count(ok);
    }

    /// Prints the counts; true if every check passed and the test ran the runs it expected
    bool passed(int expected_runs) const
    {
        std::cout << std::endl << n_runs << " runs (expected " << expected_runs << "), "
                  << n_checks << " checks, " << n_failed << " failed in "
                  << n_failed_runs << " runs" << std::endl;
        return n_failed == 0 && n_runs == expected_runs;
    }

private:
    std::string run_name;
    int failed_before_run = 0;

    bool count(bool ok)
    {
        ++n_checks;
        if (!ok)
            ++n_failed;
        return ok;
    }
};

/// sqrt(max_i |a[i] - b[i]|^2), the distance of values from stored ones; NaN if any difference is NaN
template <typename A, typename B>
double max_distance(const A* a, const B* b, int n)
{
    std::vector<double> diff_sq(n);
    for (int i = 0; i < n; i++)
        diff_sq[i] = std::pow(std::abs(a[i] - b[i]), 2);
    return std::sqrt(max_or_nan(diff_sq));
}

/// Spread of a result over the platforms and modes of a case: sqrt|E[x^2] - E[x]^2|
inline double standard_deviation(const std::vector<double>& values)
{
    double mean = std::accumulate(values.begin(), values.end(), 0.0)/values.size();
    double sq_sum = std::inner_product(values.begin(), values.end(), values.begin(), 0.0);
    return std::sqrt(std::abs(sq_sum / values.size() - mean * mean));
}

#endif
