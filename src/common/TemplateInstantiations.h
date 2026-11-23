/**
 * @file TemplateInstantiations.h
 * @brief Centralized template instantiation macros for consistent instantiation.
 *
 * This header provides macros to ensure consistent template instantiation
 * across the codebase. All template classes should use these macros at the
 * end of their implementation files.
 *
 * **Supported Types:**
 * - double: Real-valued fields (standard SCFT/L-FTS)
 * - std::complex<double>: Complex-valued fields (advanced applications)
 *
 * **Usage:**
 *
 * At the end of each .cpp/.cu implementation file:
 * @code
 * #include "TemplateInstantiations.h"
 * INSTANTIATE_CLASS(MyTemplateClass);
 * @endcode
 *
 * **Maintenance:**
 *
 * To add a new supported type (e.g., float):
 * 1. Add to INSTANTIATE_CLASS macro
 * 2. Update this documentation
 * 3. Rebuild all affected targets
 *
 * @see PropagatorComputation for usage example
 */

#ifndef TEMPLATE_INSTANTIATIONS_H
#define TEMPLATE_INSTANTIATIONS_H

#include <complex>

/**
 * @brief Instantiate a single-parameter template class for standard types.
 *
 * Creates explicit instantiations for:
 * - ClassName<double>
 * - ClassName<std::complex<double>>
 *
 * @param ClassName The template class to instantiate
 *
 * Example:
 * @code
 * // At end of MyClass.cpp
 * INSTANTIATE_CLASS(MyClass);
 * // Equivalent to:
 * // template class MyClass<double>;
 * // template class MyClass<std::complex<double>>;
 * @endcode
 */
#define INSTANTIATE_CLASS(ClassName) \
    template class ClassName<double>; \
    template class ClassName<std::complex<double>>;

#endif // TEMPLATE_INSTANTIATIONS_H
