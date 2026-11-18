/**
 * @file AlignedScratch.h
 * @brief Scratch arrays of doubles that start on a 64-byte boundary.
 *
 * MKL may pick its code by the alignment of the arrays it is given, and a
 * std::vector of each thread starts at its own offset from a 64-byte boundary,
 * so the same transform could run different code, and round differently, on
 * different threads. The per-thread scratch of the MKL CrysFFT engines lives in
 * these arrays instead, as that of MakhoulTransform does.
 */

#ifndef ALIGNED_SCRATCH_H_
#define ALIGNED_SCRATCH_H_

#include <cstddef>
#include <cstdint>
#include <vector>

class AlignedScratch
{
private:
    std::vector<double> storage_;
    std::size_t size_ = 0;

public:
    /// At least n doubles from a 64-byte boundary; the values are kept while n does not exceed size()
    double* get(std::size_t n)
    {
        constexpr std::size_t line = 64 / sizeof(double);
        if (n > size_)
        {
            storage_.resize(n + line);
            size_ = n;
        }
        const auto address = reinterpret_cast<std::uintptr_t>(storage_.data());
        return storage_.data() + (64 - address % 64) % 64 / sizeof(double);
    }

    /// The most doubles get() has been asked for
    std::size_t size() const { return size_; }
};

#endif  // ALIGNED_SCRATCH_H_
