/**
 * @file polymerfts_core.cpp
 * @brief Python bindings for the polymer field theory simulation library.
 *
 * This file uses pybind11 to expose C++ classes and functions to Python,
 * creating the `_core` module that is imported by the Python layer.
 *
 * **Exposed Classes:**
 *
 * - ComputationBox (Real/Complex): Grid and FFT operations
 * - Molecules: Polymer architecture definitions
 * - Polymer: Individual polymer chain data
 * - PropagatorComputation (Real/Complex): Propagator solving
 * - AndersonMixing (Real/Complex): Field mixing acceleration
 * - FourierTransform (Real/Complex): Fourier multipliers of whole fields
 * - AbstractFactory (Real/Complex): Platform-specific object creation
 * - PlatformSelector: Platform detection and factory creation
 * - PropagatorComputationOptimizer: Propagator computation planning
 *
 * **Type Support:**
 *
 * Classes are templated for both real (double) and complex (std::complex<double>)
 * types. Python class names use suffixes: _Real, _Complex.
 *
 * **NumPy Integration:**
 *
 * An input array is read in place when it is C-contiguous and of the expected
 * dtype; other arrays (strided views, float32, ...) are first copied into a
 * converted temporary (see array_in). Functions that modify an array in place
 * (ComputationBox.zero_mean) reject arrays
 * that would need such a copy. Real inputs refuse complex arrays with a ValueError
 * (real_array). Results are returned as new NumPy arrays.
 * Array sizes are checked against the grid.
 *
 * **Usage from Python:**
 *
 * @code{.py}
 * from polymerfts import _core
 *
 * # Create platform factory
 * factory = _core.PlatformSelector.create_factory("cuda", reduce_memory=False)
 *
 * # Create computation objects
 * cb = factory.create_computation_box(nx=[32,32,32], lx=[4.0,4.0,4.0])
 * molecules = factory.create_molecules_information("continuous", 0.1, {"A": 1.0, "B": 1.0})
 * @endcode
 *
 * @see scft.py for high-level SCFT interface
 * @see lfts.py for Langevin dynamics interface
 */

#include <tuple>
#include <map>
#include <cstring>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <optional>
#include <type_traits>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include "BuildInfo.h"
#include "NumericalMethods.h"
#ifdef USE_CPU_FFTW
#include <fftw3.h>
#endif
#ifdef USE_CPU_MKL
#include <mkl.h>
#endif
#ifdef USE_CUDA
#include "CudaCommon.h"
#include <cuda_runtime_api.h>
#include <cufft.h>
#endif

#include "GridLimits.h"
#include "Polymer.h"
#include "PropagatorComputationOptimizer.h"
#include "ComputationBox.h"
#include "MinimalFormAlias.h"
#include "PropagatorComputation.h"
#include "AndersonMixing.h"
#include "AbstractFactory.h"
#include "PlatformSelector.h"
#include "SpaceGroup.h"
#include "Exception.h"
#include "ValidationUtils.h"

namespace py = pybind11;

/**
 * @brief A NumPy array of real values (float64) given to the bindings.
 *
 * NumPy converts a complex array to float64 by dropping its imaginary part, with
 * only a ComplexWarning. A real_array refuses complex arrays instead, with a
 * ValueError, before any conversion: this one check covers every real input
 * (fields of a real solver, its Anderson mixing and Fourier transforms, masks and
 * space-group conversions), whether it is a function argument (the caster below)
 * or converted with cast<>(). Integer and boolean arrays convert as before.
 *
 * FieldTypeHint: the argument takes complex values in a solver of complex fields
 * (the fields and their Fourier coefficients), so the message names field_type;
 * masks and space-group fields are real for every field type.
 */
namespace
{
template <typename T, int Flags, bool FieldTypeHint>
class real_array : public py::array_t<T, Flags>
{
public:
    real_array() = default;
    explicit real_array(py::array_t<T, Flags>&& values) : py::array_t<T, Flags>(std::move(values)) {}
    // What handle.cast<real_array>() calls: pybind11 constructs a py::object subclass from the object
    explicit real_array(const py::object& values) : py::array_t<T, Flags>(real_values(values)) {}

    /// The values themselves; a ValueError if they are complex
    static const py::object& real_values(const py::object& values)
    {
        // The array itself (no copy), or what NumPy makes of a list; empty if neither
        const py::array as_array = py::array::ensure(values);
        if (as_array && as_array.dtype().kind() == 'c')
        {
            std::string message = "This argument takes real values, got a complex array (a conversion would "
                "drop its imaginary part).";
            if constexpr (FieldTypeHint)
                message += " Complex fields need a solver created with field_type=\"complex\".";
            throw py::value_error(message);
        }
        return values;
    }
};
}  // namespace (no wider visibility than pybind11's array_t, hidden in a module)

// Specialized like pybind11's caster of array_t (pyobject_caster; the type_caster of a
// py::object subclass resolves to it)
namespace pybind11::detail
{
template <typename T, int Flags, bool FieldTypeHint>
struct pyobject_caster<real_array<T, Flags, FieldTypeHint>>
{
    using type = real_array<T, Flags, FieldTypeHint>;
    using array_type = array_t<T, Flags>;
    PYBIND11_TYPE_CASTER(type, handle_type_name<array_type>::name);

    // Refused in both passes of the overload resolution: no overload takes a complex
    // array in place of a real one, and a noconvert (in-place) argument gets the same error
    bool load(handle src, bool convert)
    {
        type::real_values(reinterpret_borrow<object>(src));
        // As pybind11 loads array_t<T, Flags>
        if (!convert && !array_type::check_(src))
            return false;
        array_type converted = array_type::ensure(src);
        if (!converted)
            return false;
        value = type(std::move(converted));
        return true;
    }

    static handle cast(const type& src, return_value_policy, handle)
    {
        return src.inc_ref();
    }
};
}  // namespace pybind11::detail

// Input arrays are taken C-contiguous and in the expected dtype: a strided
// NumPy view or a float32 array is copied/cast, never read as if contiguous.
// Real ones refuse complex arrays (real_array). array_in takes the fields of a
// solver, which a solver of complex fields takes complex (the message names
// field_type); real_in takes masks and space-group fields, real for every field type.
template <typename T>
using array_in = std::conditional_t<std::is_same_v<std::remove_const_t<T>, double>,
    real_array<T, py::array::c_style | py::array::forcecast, true>,
    py::array_t<T, py::array::c_style | py::array::forcecast>>;
using real_in = real_array<double, py::array::c_style | py::array::forcecast, false>;

// Arrays modified in place: taken as they are (with py::arg().noconvert()), so the
// function can refuse those that would need a copy. Real ones refuse complex arrays.
template <typename T>
using array_inout = std::conditional_t<std::is_same_v<T, double>, real_array<T, 0, true>, py::array_t<T, 0>>;

/**
 * @brief Helper for method overload resolution in pybind11.
 *
 * Used to disambiguate overloaded C++ methods when binding to Python.
 */
template <typename... Args>
using overload_cast_ = py::detail::overload_cast_impl<Args...>;

/**
 * @brief The data of an input array, which must hold n values.
 *
 * @param what      The input, for the error message (e.g. "input w[A]")
 * @param size_name The size it must have (e.g. "n_basis", the size of the basis of the box)
 */
template <typename T, int Flags>
T* checked_data(const py::array_t<T, Flags>& array, long long n, const std::string& what, const std::string& size_name)
{
    const py::buffer_info buf = array.request();
    if (buf.size != n)
        throw_with_line_number("Size of " + what + " (" + std::to_string(buf.size) + ") and '" + size_name
            + "' (" + std::to_string(n) + ") must match.");
    return static_cast<T*>(buf.ptr);
}

/**
 * @brief A new array of fields in the basis of a computation (reduced with a space group), shaped
 *        lead + (n_basis,), which fill(data) fills.
 */
template <typename T, typename Fill>
py::array_t<T> basis_fields(PropagatorComputation<T>& obj, std::vector<py::ssize_t> lead, Fill&& fill)
{
    lead.push_back(obj.get_cb()->get_n_basis());
    py::array_t<T> result(lead);
    fill(result.mutable_data());
    return result;
}

/**
 * @brief Bind ComputationBox<T> class to Python.
 *
 * Creates Python class "ComputationBox{Real|Complex}" with methods:
 * - get_nx(): Grid points per dimension
 * - get_lx(): Box lengths
 * - get_dx(): Grid spacing
 * - get_dv(): Volume element of a grid point with the mask; get_dv_base(): without it
 * - get_total_grid(): Total grid points
 * - get_volume(): Total box volume
 * - get_angles_degrees(), get_lattice_vectors(), is_orthogonal(), get_recip_metric(): Cell geometry
 * - set_lx(): Update box dimensions
 * - set_lattice_parameters(): Update box dimensions and angles
 * - get_space_group(), get_n_basis(), get_orbit_counts(): Reduced basis of the space group given at creation
 * - integral(), mean(): Volume integral / average of a field
 * - inner_product(), inner_product_inverse_weight(), multi_inner_product(): Inner products
 * - zero_mean(): Subtract the mean in place (C-contiguous array required)
 *
 * @tparam T Field type (double or std::complex<double>)
 * @param m Python module to add class to
 * @param type_name Suffix for class name ("Real" or "Complex")
 */
template<typename T>
void bind_computation_box(py::module &m, const std::string &type_name) {
    std::string class_name = "ComputationBox" + type_name;
    
    py::class_<ComputationBox<T>>(m, class_name.c_str())
        .def("get_nx", py::overload_cast<>(&ComputationBox<T>::get_nx, py::const_))
        .def("get_nx", py::overload_cast<int>(&ComputationBox<T>::get_nx, py::const_))
        .def("get_lx", py::overload_cast<>(&ComputationBox<T>::get_lx, py::const_))
        .def("get_lx", py::overload_cast<int>(&ComputationBox<T>::get_lx, py::const_))
        .def("get_dx", py::overload_cast<>(&ComputationBox<T>::get_dx, py::const_))
        .def("get_dx", py::overload_cast<int>(&ComputationBox<T>::get_dx, py::const_))
        .def("get_dv", [](const ComputationBox<T>& obj, int i)
        {
            validation::require_index(i, obj.get_total_grid(), "Index");
            return obj.get_dv(i);
        })
        .def("get_dv_base", &ComputationBox<T>::get_dv_base)
        .def("get_total_grid", &ComputationBox<T>::get_total_grid)
        .def("get_volume", &ComputationBox<T>::get_volume)
        .def("get_accessible_volume", &ComputationBox<T>::get_accessible_volume)
        .def("get_angles_degrees", &ComputationBox<T>::get_angles_degrees)
        .def("get_lattice_vectors", [](const ComputationBox<T>& obj)
        {
            // Rows a, b, c in Cartesian coordinates (a along x, b in the xy plane)
            const std::array<double, 9>& v = obj.get_lattice_vec();
            return std::vector<std::vector<double>>{{v[0], v[1], v[2]}, {v[3], v[4], v[5]}, {v[6], v[7], v[8]}};
        })
        .def("is_orthogonal", &ComputationBox<T>::is_orthogonal)
        .def("get_recip_metric", &ComputationBox<T>::get_recip_metric)
        .def("set_lx", &ComputationBox<T>::set_lx)
        .def("set_lattice_parameters", overload_cast_<std::vector<double>, std::vector<double>>()(&ComputationBox<T>::set_lattice_parameters))
        // The reduced basis of the space group given at creation
        .def("get_space_group", &ComputationBox<T>::get_space_group, py::return_value_policy::reference)  // kept alive by the box (keep_alive at creation)
        .def("get_n_basis", &ComputationBox<T>::get_n_basis)
        .def("get_orbit_counts", &ComputationBox<T>::get_orbit_counts)
        // Field operations: use get_n_basis() for size (n_reduced if space group set, else total_grid)
        .def("integral", [](ComputationBox<T>& obj, array_in<T> g)
        {
            return obj.integral(checked_data(g, obj.get_n_basis(), "input", "n_basis"));
        })
        .def("mean", [](ComputationBox<T>& obj, array_in<T> g)
        {
            return obj.mean(checked_data(g, obj.get_n_basis(), "input", "n_basis"));
        })
        .def("inner_product", [](ComputationBox<T>& obj, array_in<T> g, array_in<T> h)
        {
            const int N = obj.get_n_basis();
            return obj.inner_product(checked_data(g, N, "input g", "n_basis"), checked_data(h, N, "input h", "n_basis"));
        })
        .def("inner_product_inverse_weight", [](ComputationBox<T>& obj, array_in<T> g, array_in<T> h, array_in<T> w)
        {
            const int N = obj.get_n_basis();
            return obj.inner_product_inverse_weight(checked_data(g, N, "input g", "n_basis"),
                checked_data(h, N, "input h", "n_basis"), checked_data(w, N, "input w", "n_basis"));
        })
        .def("multi_inner_product", [](ComputationBox<T>& obj, int n_comp, array_in<T> g, array_in<T> h)
        {
            const long long n_total = static_cast<long long>(n_comp) * obj.get_n_basis();
            return obj.multi_inner_product(n_comp, checked_data(g, n_total, "g", "n_comp * n_basis"),
                                           checked_data(h, n_total, "h", "n_comp * n_basis"));
        })
        .def("zero_mean", [](ComputationBox<T>& obj, array_inout<T> g)
        {
            // Modified in place: a converted copy would leave the caller's array
            // unchanged, so the array must already be C-contiguous and of the field type
            // (a wrong dtype is rejected by pybind11 before the call, a complex array
            // of real fields by real_array), and writable (a read-only memory map would
            // crash, an immutable buffer would change).
            if (!(g.flags() & py::array::c_style) || !g.writeable())
                throw_with_line_number("zero_mean modifies the array in place; pass a writable C-contiguous array.");
            obj.zero_mean(checked_data(g, obj.get_n_basis(), "input", "n_basis"));
        }, py::arg("g").noconvert());
}

/**
 * @brief Bind PropagatorComputation<T> class to Python.
 *
 * Creates Python class "PropagatorComputation_{Real|Complex}" with methods:
 * - update_laplacian_operator(): Refresh operators after box change
 * - get_cb(): Get ComputationBox (use cb.get_n_basis(), cb.get_total_grid(), etc.)
 * - compute_propagators(w_input, q_init=None): Solve propagator equations
 * - advance_propagator_single_segment(q_in, p, v, u): Single step for block (v,u) of polymer p
 *   (internal; use compute_propagators() and get_chain_propagator() instead)
 * - compute_concentrations(): Calculate segment densities
 * - get_total_concentration(monomer_type): Total density by monomer
 * - get_total_concentration(polymer, monomer_type): Per-polymer density
 * - get_total_concentration_gce(fugacity, polymer, monomer_type): Grand canonical
 * - get_block_concentration(polymer): Block-resolved density
 * - get_total_partition(polymer): Partition function
 * - get_solvent_partition(s): Solvent partition
 * - get_solvent_concentration(s): Solvent density
 * - get_chain_propagator(polymer, v, u, n): Extract propagator at step n
 * - compute_stress(): Calculate stress tensor
 * - compute_concentrations_and_stress(): Both, in one recomputation with reduced memory
 * - get_stress(): Retrieve computed stress
 * - get_stress_gce(): Grand canonical stress
 * - compute_partition_errors(): Relative partition function error of each polymer
 * - check_total_partition(tolerance): Whether every error is within the tolerance
 * - set_force_off_diagonal_stress(): Cross-term stress for orthogonal boxes
 *
 * When space group is set:
 * - compute_propagators() accepts reduced basis w (size: n_reduced)
 * - get_total_concentration() returns reduced basis phi (size: n_reduced)
 *
 * @tparam T Field type (double or std::complex<double>)
 * @param m Python module
 * @param type_name Suffix ("Real" or "Complex")
 */
template<typename T>
void bind_propagator_computation(py::module &m, const std::string &type_name) {
    std::string class_name = "PropagatorComputation_" + type_name;
    
    py::class_<PropagatorComputation<T>>(m, class_name.c_str())
        .def("update_laplacian_operator", &PropagatorComputation<T>::update_laplacian_operator)
        .def("get_cb", &PropagatorComputation<T>::get_cb, py::return_value_policy::reference_internal)
        // compute_propagators: accepts reduced basis w (n_grid) when space group is set,
        // otherwise accepts full grid w (total_grid)
        .def("compute_propagators", [](PropagatorComputation<T>& obj, std::map<std::string, array_in<const T>> w_input, py::object q_init)
        {
            // The casts own the dtype/contiguity conversions of their arrays until the
            // computation has read them; a complex q_init of real fields raises the
            // ValueError of real_array, like w_input
            std::map<std::string, array_in<const T>> q_init_arrays;
            if (!q_init.is_none())
                q_init_arrays = q_init.cast<std::map<std::string, array_in<const T>>>();

            // q_init is in the same basis as w (reduced with a space group), the layout the computations copy
            const int N = obj.get_cb()->get_n_basis();
            std::map<std::string, const T*> w, q;
            for (const auto& [monomer_type, values] : w_input)
                w[monomer_type] = checked_data(values, N, "input w[" + monomer_type + "]", "n_basis");
            for (const auto& [label, values] : q_init_arrays)
                q[label] = checked_data(values, N, "input q[" + label + "]", "n_basis");
            obj.compute_propagators(w, q);
        }, py::arg("w_input"), py::arg("q_init") = py::none())
        .def("advance_propagator_single_segment", [](PropagatorComputation<T>& obj, array_in<T> q_in, int p, int v, int u)
        {
            // In the basis of the box (reduced with a space group), as in compute_propagators
            T* q = checked_data(q_in, obj.get_cb()->get_n_basis(), "input q_in", "n_basis");
            return basis_fields(obj, {}, [&](T* q_out) { obj.advance_propagator_single_segment(q, q_out, p, v, u); });
        })
        .def("compute_concentrations", &PropagatorComputation<T>::compute_concentrations)
        // The getters return fields in the basis of the box: reduced with a space group, else the full grid
        .def("get_total_concentration", [](PropagatorComputation<T>& obj, std::string monomer_type)
        {
            return basis_fields(obj, {}, [&](T* phi) { obj.get_total_concentration(monomer_type, phi); });
        })
        .def("get_total_concentration", [](PropagatorComputation<T>& obj, int polymer, std::string monomer_type)
        {
            return basis_fields(obj, {}, [&](T* phi) { obj.get_total_concentration(polymer, monomer_type, phi); });
        })
        .def("get_total_concentration_gce", [](PropagatorComputation<T>& obj, double fugacity, int polymer, std::string monomer_type)
        {
            return basis_fields(obj, {}, [&](T* phi) { obj.get_total_concentration_gce(fugacity, polymer, monomer_type, phi); });
        })
        .def("get_block_concentration", [](PropagatorComputation<T>& obj, int polymer)
        {
            return basis_fields(obj, {obj.get_n_blocks(polymer)}, [&](T* phi) { obj.get_block_concentration(polymer, phi); });
        })
        .def("get_total_partition", &PropagatorComputation<T>::get_total_partition)
        .def("get_solvent_partition", &PropagatorComputation<T>::get_solvent_partition)
        .def("get_solvent_concentration", [](PropagatorComputation<T>& obj, int s)
        {
            return basis_fields(obj, {}, [&](T* phi) { obj.get_solvent_concentration(s, phi); });
        })
        .def("get_chain_propagator", [](PropagatorComputation<T>& obj, int polymer, int v, int u, int n)
        {
            return basis_fields(obj, {}, [&](T* q) { obj.get_chain_propagator(q, polymer, v, u, n); });
        })
        .def("compute_stress", &PropagatorComputation<T>::compute_stress)
        .def("compute_concentrations_and_stress", &PropagatorComputation<T>::compute_concentrations_and_stress,
            "compute_concentrations() and compute_stress(), with one recomputation of the propagators in reduced memory")
        .def("set_force_off_diagonal_stress", &PropagatorComputation<T>::set_force_off_diagonal_stress,
            "Force cross-term (V12/V13/V23) stress accumulation even for orthogonal boxes. "
            "Needed when angle optimization starts at exactly 90 degree angles.")
        .def("get_stress", &PropagatorComputation<T>::get_stress)
        .def("get_stress_gce", &PropagatorComputation<T>::get_stress_gce)
        .def("compute_partition_errors", &PropagatorComputation<T>::compute_partition_errors,
            "Relative partition function error of each polymer (nothing is printed)")
        .def("check_total_partition", &PropagatorComputation<T>::check_total_partition,
            py::arg("tolerance") = PARTITION_TOLERANCE,
            "Whether every error of compute_partition_errors() is finite and at most tolerance");
}

/**
 * @brief Bind AndersonMixing<T> class to Python.
 *
 * Creates Python class "AndersonMixing_{Real|Complex}" with methods:
 * - reset_count(): Clear history and restart mixing
 * - calculate_new_fields(w_current, w_deriv, old_error, error): Compute mixed field
 *
 * The calculate_new_fields method takes NumPy arrays for:
 * - w_current: Current field values
 * - w_deriv: Field residual (w_new - w_current from SCFT equations)
 * - old_error_level, error_level: Error norms for adaptive mixing
 *
 * Returns: w_new NumPy array with the mixed field
 *
 * @tparam T Field type (double or std::complex<double>)
 * @param m Python module
 * @param type_name Suffix ("Real" or "Complex")
 */
template<typename T>
void bind_anderson_mixing(py::module &m, const std::string &type_name) {
    std::string class_name = "AndersonMixing_" + type_name;
    
    py::class_<AndersonMixing<T>>(m, class_name.c_str())
        .def("reset_count", &AndersonMixing<T>::reset_count)
        .def("calculate_new_fields", [](AndersonMixing<T> &obj,
                array_in<T> w_current, array_in<T> w_deriv,
                double old_error_level, double error_level)
        {
            const int n_var = obj.get_n_var();
            T* current = checked_data(w_current, n_var, "input w_current", "n_var");
            T* deriv = checked_data(w_deriv, n_var, "input w_deriv", "n_var");
            py::array_t<T> w_new(n_var);
            obj.calculate_new_fields(w_new.mutable_data(), current, deriv, old_error_level, error_level);
            return w_new;
        });
}

/**
 * @brief Bind FourierTransform<T> and its FourierMultiplier<T> handles to Python.
 *
 * Creates the Python classes "FourierTransform_{Real|Complex}" and
 * "FourierMultiplier_{Real|Complex}":
 * - coefficient_shape, complex_coefficients: Layout of the coefficients of a field
 * - field_type: "real" or "complex", the fields the transform takes; periodic: every axis periodic
 *   (FFT); mixed: periodic and wall axes (one FFT of the grid with Makhoul's reordering on the walls)
 * - k_squared(): |k|² in the coefficient shape, from the current box
 * - alias_moments(): <v_a v_b>, v = g⁻¹u, over the aliases u tied for the least |k|²
 *   (periodic boxes), shape (dim, dim) + coefficient_shape
 * - set_multiplier(values): Store a multiplier of shape coefficient_shape
 *   (diagonal: applied to each field) or (n_out, n_in) + coefficient_shape
 * - apply([(multiplier, fields), ...]): backward(Σ M · forward(fields))
 * - forward(fields), backward(coefficients): Transforms of one or more fields
 *
 * Fields are flat (n_grid,) or (n_fields, n_grid); the output of apply() is flat
 * when every input is flat and there is one output field.
 *
 * @tparam T Field type (double or std::complex<double>)
 * @param m Python module
 * @param type_name Suffix ("Real" or "Complex")
 */
template<typename T>
void bind_fourier_transform(py::module &m, const std::string &type_name)
{
    // Number of fields of a flat (n_grid,) or (n_fields, n_grid) array
    auto field_count = [](const py::buffer_info& buf, int n_grid) -> int
    {
        if ((buf.ndim != 1 && buf.ndim != 2) || buf.shape[buf.ndim - 1] != n_grid || buf.size == 0)
            throw_with_line_number("Fields must be shaped (" + std::to_string(n_grid) + ",) or (n_fields, "
                + std::to_string(n_grid) + ").");
        return buf.ndim == 1 ? 1 : static_cast<int>(buf.shape[0]);
    };
    // Number of fields of an array of coefficients: coefficient_shape or (n_fields,) + coefficient_shape
    auto coefficient_count = [](const py::buffer_info& buf, const std::vector<int>& shape) -> int
    {
        const int lead = static_cast<int>(buf.ndim) - static_cast<int>(shape.size());
        bool ok = (lead == 0 || lead == 1) && buf.size > 0;
        for (size_t d = 0; ok && d < shape.size(); ++d)
            ok = buf.shape[lead + d] == shape[d];
        if (!ok)
            throw_with_line_number("Coefficients must have the coefficient shape, optionally after the number of fields.");
        return lead == 0 ? 1 : static_cast<int>(buf.shape[0]);
    };
    auto shaped = [](std::vector<py::ssize_t> lead, const std::vector<int>& shape)
    {
        for (int n : shape)
            lead.push_back(n);
        return lead;
    };
    constexpr bool real_fields = std::is_same_v<T, double>;

    py::class_<FourierMultiplier<T>>(m, ("FourierMultiplier_" + type_name).c_str())
        .def_property_readonly("n_out", &FourierMultiplier<T>::get_n_out)
        .def_property_readonly("n_in", &FourierMultiplier<T>::get_n_in)
        .def_property_readonly("diagonal", &FourierMultiplier<T>::is_diagonal);

    py::class_<FourierTransform<T>>(m, ("FourierTransform_" + type_name).c_str())
        .def_property_readonly("coefficient_shape", [](FourierTransform<T>& obj)
        {
            return py::tuple(py::cast(obj.get_coefficient_shape()));
        })
        .def_property_readonly("complex_coefficients", &FourierTransform<T>::has_complex_coefficients)
        .def_property_readonly("field_type", [](FourierTransform<T>&) { return real_fields ? "real" : "complex"; })
        .def_property_readonly("periodic", &FourierTransform<T>::is_periodic)   // every axis periodic
        .def_property_readonly("mixed", &FourierTransform<T>::is_mixed)         // periodic and wall axes
        .def("k_squared", [](FourierTransform<T>& obj)
        {
            const std::vector<double> k_sq = obj.get_k_squared();
            py::array_t<double> result(obj.get_coefficient_shape());
            std::copy(k_sq.begin(), k_sq.end(), result.mutable_data());
            return result;
        })
        .def("alias_moments", [](FourierTransform<T>& obj)
        {
            const std::vector<double> moments = obj.get_alias_moments();
            const int dim = static_cast<int>(obj.get_coefficient_shape().size());
            std::vector<py::ssize_t> shape = {dim, dim};
            for (int n : obj.get_coefficient_shape())
                shape.push_back(n);
            py::array_t<double> result(shape);
            std::copy(moments.begin(), moments.end(), result.mutable_data());
            return result;
        })
        // The handle keeps its transform alive (it is used only with that transform)
        .def("set_multiplier", [](FourierTransform<T>& obj, array_in<std::complex<double>> values)
        {
            py::buffer_info buf = values.request();
            const std::vector<int>& shape = obj.get_coefficient_shape();
            const int lead = static_cast<int>(buf.ndim) - static_cast<int>(shape.size());
            bool ok = lead == 0 || lead == 2;
            for (size_t d = 0; ok && d < shape.size(); ++d)
                ok = buf.shape[lead + d] == shape[d];
            if (!ok || buf.size == 0)
                throw_with_line_number("A multiplier is shaped like the coefficients (diagonal) "
                    "or (n_out, n_in) + coefficient_shape.");
            const bool diagonal = lead == 0;
            const int n_out = diagonal ? 1 : static_cast<int>(buf.shape[0]);
            const int n_in = diagonal ? 1 : static_cast<int>(buf.shape[1]);
            return obj.create_multiplier(static_cast<const std::complex<double>*>(buf.ptr), n_out, n_in, diagonal);
        }, py::arg("values"), py::keep_alive<0, 1>())
        .def("apply", [field_count](FourierTransform<T>& obj, py::list terms)
        {
            const int n_grid = obj.get_n_grid();
            std::vector<array_in<T>> inputs;   // keeps converted inputs alive
            std::vector<FourierTerm<T>> fourier_terms;
            bool all_flat = true;
            for (py::handle item : terms)
            {
                py::tuple pair = py::cast<py::tuple>(item);
                if (pair.size() != 2)
                    throw_with_line_number("Each term is a (multiplier, fields) pair.");
                const FourierMultiplier<T>* multiplier = pair[0].cast<const FourierMultiplier<T>*>();
                if (multiplier == nullptr)
                    throw_with_line_number("Each term needs a multiplier made by set_multiplier().");
                inputs.push_back(pair[1].cast<array_in<T>>());
                py::buffer_info buf = inputs.back().request();
                const int n_fields = field_count(buf, n_grid);
                all_flat = all_flat && buf.ndim == 1;
                fourier_terms.push_back({multiplier, static_cast<const T*>(buf.ptr), n_fields});
            }
            if (fourier_terms.empty())
                throw_with_line_number("apply() needs at least one term.");
            const FourierTerm<T>& first = fourier_terms[0];
            const int n_out = first.multiplier->is_diagonal() ? first.n_fields : first.multiplier->get_n_out();
            py::array_t<T> result = (all_flat && n_out == 1)
                ? py::array_t<T>(n_grid) : py::array_t<T>({static_cast<py::ssize_t>(n_out), static_cast<py::ssize_t>(n_grid)});
            obj.apply(fourier_terms, n_out, result.mutable_data());
            return result;
        }, py::arg("terms"))
        .def("forward", [field_count, shaped](FourierTransform<T>& obj, py::object values) -> py::array
        {
            array_in<T> fields = values.cast<array_in<T>>();
            py::buffer_info buf = fields.request();
            const int n_fields = field_count(buf, obj.get_n_grid());
            const std::vector<py::ssize_t> lead = buf.ndim == 1 ? std::vector<py::ssize_t>{} : std::vector<py::ssize_t>{n_fields};
            const std::vector<py::ssize_t> shape = shaped(lead, obj.get_coefficient_shape());
            if (obj.has_complex_coefficients())
            {
                py::array_t<std::complex<double>> result(shape);
                obj.forward(static_cast<const T*>(buf.ptr), n_fields, reinterpret_cast<double*>(result.mutable_data()));
                return result;
            }
            py::array_t<double> result(shape);
            obj.forward(static_cast<const T*>(buf.ptr), n_fields, result.mutable_data());
            return result;
        }, py::arg("fields"))
        .def("backward", [coefficient_count](FourierTransform<T>& obj, py::object coefficients)
        {
            const int n_grid = obj.get_n_grid();
            auto run = [&](auto input, const double* values)
            {
                py::buffer_info buf = input.request();
                const int n_fields = coefficient_count(buf, obj.get_coefficient_shape());
                const bool flat = buf.ndim == static_cast<py::ssize_t>(obj.get_coefficient_shape().size());
                py::array_t<T> result = flat ? py::array_t<T>(n_grid)
                    : py::array_t<T>({static_cast<py::ssize_t>(n_fields), static_cast<py::ssize_t>(n_grid)});
                obj.backward(values, n_fields, result.mutable_data());
                return result;
            };
            if (obj.has_complex_coefficients())
            {
                array_in<std::complex<double>> input = coefficients.cast<array_in<std::complex<double>>>();
                return run(input, reinterpret_cast<const double*>(input.data()));
            }
            array_in<double> input = coefficients.cast<array_in<double>>();
            return run(input, input.data());
        }, py::arg("coefficients"));
}

/**
 * @brief Bind AbstractFactory<T> class to Python.
 *
 * Creates Python class "AbstractFactory_{Real|Complex}" with factory methods:
 * - create_computation_box(nx, lx, angles=None, bc=None, mask=None, space_group=None): Create ComputationBox
 * - create_molecules_information(): Create Molecules container
 * - create_propagator_computation_optimizer(): Create optimizer
 * - create_propagator_computation(): Create propagator solver
 * - create_anderson_mixing(): Create Anderson mixing
 * - create_fourier_transform(cb, bc): Create a Fourier transform (one transform per axis)
 * - display_info(): Print platform information
 *
 * The factory pattern ensures all created objects are compatible with
 * the selected platform (CPU-FFTW or CUDA).
 *
 * @tparam T Field type (double or std::complex<double>)
 * @param m Python module
 * @param type_name Suffix ("Real" or "Complex")
 */
template<typename T>
void bind_abstract_factory(py::module &m, const std::string &type_name)
{
    std::string class_name = "AbstractFactory_" + type_name;

    // Positions of py::keep_alive<nurse, patient>: 0 is the created object, 1 the factory and 2...
    // the arguments; that of the space group of create_computation_box and those of
    // create_propagator_computation:
    constexpr size_t CREATED = 0, BOX_SPACE_GROUP = 7, CB = 2, MOLECULES = 3, OPTIMIZER = 4;

    py::class_<AbstractFactory<T>>(m, class_name.c_str())
        .def("create_computation_box", [](
            AbstractFactory<T>& obj,
            std::vector<int> nx, std::vector<double> lx,
            std::optional<std::vector<double>> angles,
            std::optional<std::vector<std::string>> bc,
            std::optional<real_in> mask,  // real for both field types
            SpaceGroup* space_group)
        {
            const int M = checked_grid_points(nx);
            const double* mask_data = mask ? checked_data(*mask, M, "input mask", "total_grid") : nullptr;
            return obj.create_computation_box(nx, lx, bc.value_or(std::vector<std::string>{}),
                                              angles.value_or(std::vector<double>{}), mask_data, space_group);
        }, py::arg("nx"), py::arg("lx"), py::arg("angles") = py::none(), py::arg("bc") = py::none(), py::arg("mask") = py::none(),
           py::arg("space_group") = nullptr,
           py::keep_alive<CREATED, BOX_SPACE_GROUP>())  // the box keeps a pointer to its space group
        .def("create_molecules_information", &AbstractFactory<T>::create_molecules_information)
        // keep_alive: the C++ objects hold raw pointers to their arguments, so the
        // Python wrappers of those arguments must outlive the created object
        .def("create_propagator_computation_optimizer", &AbstractFactory<T>::create_propagator_computation_optimizer,
            py::arg("molecules").none(false), py::arg("aggregate_propagator_computation"),
            py::keep_alive<CREATED, 2>())  // molecules
        .def("create_propagator_computation", &AbstractFactory<T>::create_propagator_computation,
            py::arg("cb"), py::arg("molecules"), py::arg("propagator_computation_optimizer"),
            py::arg("numerical_method"),
            py::keep_alive<CREATED, CB>(), py::keep_alive<CREATED, MOLECULES>(), py::keep_alive<CREATED, OPTIMIZER>())
        .def("create_anderson_mixing", [](AbstractFactory<T>& obj, long long n_var, int max_hist,
                                          double start_error, double mix_min, double mix_init)
        {
            // A Python size can exceed int: check it before narrowing
            check_anderson_variables(n_var);
            return obj.create_anderson_mixing(static_cast<int>(n_var), max_hist, start_error, mix_min, mix_init);
        }, py::arg("n_var"), py::arg("max_hist"), py::arg("start_error"), py::arg("mix_min"), py::arg("mix_init"))
        .def("create_fourier_transform", [](AbstractFactory<T>& obj, ComputationBox<T>* cb, std::vector<std::string> bc)
        {
            // One transform per axis: "periodic", "reflecting" (cosine) or "absorbing" (sine)
            std::vector<BoundaryCondition> bc_vec;
            for (std::string name : bc)
            {
                std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return std::tolower(c); });
                if (name == "periodic")
                    bc_vec.push_back(BoundaryCondition::PERIODIC);
                else if (name == "reflecting")
                    bc_vec.push_back(BoundaryCondition::REFLECTING);
                else if (name == "absorbing")
                    bc_vec.push_back(BoundaryCondition::ABSORBING);
                else
                    throw_with_line_number("Unknown transform '" + name + "': use periodic, reflecting or absorbing.");
            }
            return obj.create_fourier_transform(cb, bc_vec);
        }, py::arg("cb"), py::arg("bc"), py::keep_alive<CREATED, 2>())  // cb
        .def("display_info", &AbstractFactory<T>::display_info);
}

// Field conversions of SpaceGroup. The input holds n_fields fields of n_in values
// each, either flat (1-D) or as (n_fields, n_in); the output keeps the layout of
// the input: a 1-D input gives a 1-D output, a 2-D input an (n_fields, n_out) array.
static int sg_field_count(const py::buffer_info& in, int n_in, const char* n_in_name)
{
    if (in.ndim != 1 && in.ndim != 2)
        throw_with_line_number("Input must be flat (1-D) or shaped (n_fields, n_grid).");
    if (in.ndim == 2 && in.shape[1] != n_in)
        throw_with_line_number("The last input axis must equal " + std::string(n_in_name) +
            " (" + std::to_string(n_in) + ").");
    py::ssize_t total = 1;
    for (py::ssize_t i = 0; i < in.ndim; ++i)
        total *= in.shape[i];
    const int n_fields = static_cast<int>(total / n_in);
    if (n_fields < 1 || total != static_cast<py::ssize_t>(n_fields) * n_in)
        throw_with_line_number("Field size (" + std::to_string(total) + ") must be a multiple of " +
                               n_in_name + " (" + std::to_string(n_in) + ").");
    return n_fields;
}

static py::array_t<double> sg_output_like(const py::buffer_info& in, int n_fields, int n_out)
{
    if (in.ndim == 1)
        return py::array_t<double>(static_cast<py::ssize_t>(n_fields) * n_out);
    return py::array_t<double>({static_cast<py::ssize_t>(n_fields), static_cast<py::ssize_t>(n_out)});
}

// Converts the Python block list [[monomer_type, contour_length, v, u(, multiplicity)], ...]
static std::vector<BlockInput> to_block_inputs(const std::vector<std::vector<py::object>>& block_list)
{
    if (block_list.empty())
        throw_with_line_number("A polymer needs at least one block.");
    std::vector<BlockInput> block_inputs;
    for (size_t b = 0; b < block_list.size(); b++)
    {
        const auto& py_block = block_list[b];
        if (py_block.size() != 4 && py_block.size() != 5)
            throw_with_line_number("Block " + std::to_string(b) + " has " + std::to_string(py_block.size()) +
                " entries; a block is [monomer_type, contour_length, v, u] or [monomer_type, contour_length, v, u, multiplicity].");
        BlockInput block;
        block.monomer_type = py::cast<std::string>(py_block[0]);
        block.contour_length = py::cast<double>(py_block[1]);
        block.v = py::cast<int>(py_block[2]);
        block.u = py::cast<int>(py_block[3]);
        if (py_block.size() == 5)
            block.multiplicity = py::cast<double>(py_block[4]);
        block_inputs.push_back(block);
    }
    return block_inputs;
}

/**
 * @brief Bind Block, Polymer, Molecules and the propagator computation optimizer
 *        (ComputationEdge, ComputationBlock) to Python.
 */
static void bind_molecules(py::module& m)
{
    py::class_<Block>(m, "Block")
        .def_readonly("monomer_type", &Block::monomer_type)
        .def_readonly("n_segment", &Block::n_segment)
        .def_readonly("contour_length", &Block::contour_length)
        .def_readonly("v", &Block::v)
        .def_readonly("u", &Block::u)
        .def_readonly("n_copies", &Block::n_copies);

    py::class_<ComputationEdge>(m, "ComputationEdge")
        .def_readonly("max_n_segment", &ComputationEdge::max_n_segment)
        .def_readonly("monomer_type", &ComputationEdge::monomer_type);

    py::class_<ComputationBlock>(m, "ComputationBlock")
        .def_readonly("monomer_type", &ComputationBlock::monomer_type)
        .def_readonly("n_repeated", &ComputationBlock::n_repeated);

    py::class_<Polymer>(m, "Polymer")
        .def("get_alpha", &Polymer::get_alpha)
        .def("get_volume_fraction", &Polymer::get_volume_fraction)
        .def("get_n_blocks", &Polymer::get_n_blocks)
        .def("get_blocks", &Polymer::get_blocks)
        .def("get_block", &Polymer::get_block)
        .def("get_n_segment_total", &Polymer::get_n_segment_total);

    py::class_<Molecules>(m, "Molecules")
        .def(py::init<std::string, double, std::map<std::string, double>>())
        .def("get_model_name", &Molecules::get_model_name)
        .def("get_n_polymer_types", &Molecules::get_n_polymer_types)
        .def("add_polymer", [](Molecules& obj, double volume_fraction, std::vector<std::vector<py::object>> block_list)
        {
            obj.add_polymer(volume_fraction, to_block_inputs(block_list), {});
        })
        .def("add_polymer", [](Molecules& obj, double volume_fraction, std::vector<std::vector<py::object>> block_list, std::map<int, std::string> chain_end_to_q_init)
        {
            obj.add_polymer(volume_fraction, to_block_inputs(block_list), chain_end_to_q_init);
        })
        .def("get_polymer", &Molecules::get_polymer)
        .def("get_n_solvent_types", &Molecules::get_n_solvent_types)
        .def("get_solvent", &Molecules::get_solvent)
        .def("add_solvent", &Molecules::add_solvent)
        .def("display_architectures", &Molecules::display_architectures);

    py::class_<PropagatorComputationOptimizer>(m, "PropagatorComputationOptimizer")
        .def("get_propagator_key", &PropagatorComputationOptimizer::get_propagator_key,
             py::arg("polymer"), py::arg("v"), py::arg("u"),
             "Key of the propagator that leaves v along the block (v, u) of a polymer.")
        .def("get_computation_propagators", &PropagatorComputationOptimizer::get_computation_propagators)
        .def("get_computation_propagator", &PropagatorComputationOptimizer::get_computation_propagator)
        .def("get_computation_blocks", &PropagatorComputationOptimizer::get_computation_blocks)
        .def("display_statistics", &PropagatorComputationOptimizer::display_statistics);
}

/**
 * @brief Bind SpaceGroup to Python: reduced bases, physical bases and field conversions.
 */
static void bind_space_group(py::module& m)
{
    py::class_<SpaceGroup>(m, "SpaceGroup",
        R"doc(Space group symmetry for reduced basis representation.

Reduces computational cost by exploiting crystallographic symmetry.
Fields can be represented using only reduced-basis points.

Common space groups:
  - Im-3m (229, hall=529): BCC
  - Ia-3d (230, hall=530): Gyroid
  - Fm-3m (225, hall=523): FCC
  - P6_3/mmc (194, hall=488): HCP, PL

Example:
    >>> sg = SpaceGroup([32, 32, 32], "Im-3m", 529)
    >>> print(f"Reduction: {sg.get_total_grid() / sg.get_n_reduced_basis():.1f}x")
)doc")
        .def(py::init<std::vector<int>, int>(), py::arg("nx"), py::arg("hall_number"),
             "Create SpaceGroup from grid dimensions and Hall number.")
        .def(py::init<std::vector<int>, const std::string&, int>(),
             py::arg("nx"), py::arg("symbol"), py::arg("hall_number") = -1,
             "Create SpaceGroup from grid dimensions and ITA symbol.")
        .def_static("hall_numbers_from_symbol", &SpaceGroup::hall_numbers_from_symbol,
             "Return all Hall numbers for an ITA symbol.")
        .def("get_hall_number", &SpaceGroup::get_hall_number,
             "Hall number (1-530).")
        .def("get_spacegroup_number", &SpaceGroup::get_spacegroup_number,
             "International space group number (1-230).")
        .def("get_spacegroup_symbol", &SpaceGroup::get_spacegroup_symbol,
             "ITA short symbol.")
        .def("uses_hexagonal_axes", &SpaceGroup::uses_hexagonal_axes,
             "True for hexagonal/trigonal groups in a hexagonal-axes setting (a = b, gamma = 120); "
             "False for rhombohedral-axes settings (a = b = c, alpha = beta = gamma).")
        .def("get_crystal_system", &SpaceGroup::get_crystal_system,
             "Crystal system name.")
        .def("check_cell", &SpaceGroup::check_cell, py::arg("lx"), py::arg("angles"),
             "Raise unless the cell (lengths, angles in degrees) belongs to the crystal system of the group.")
        .def("fixed_angles", [](const SpaceGroup& sg)
        {
            py::list result;
            for (double angle : sg.fixed_angles())
                result.append(std::isnan(angle) ? py::none() : py::cast(angle));
            return result;
        }, "The angles [alpha, beta, gamma] in degrees the group fixes; None where an angle is free.")
        .def("get_nx", &SpaceGroup::get_nx,
             "Grid dimensions [nx, ny, nz].")
        .def("get_total_grid", &SpaceGroup::get_total_grid,
             "Total number of grid points.")
        .def("get_n_reduced_basis", &SpaceGroup::get_n_reduced_basis,
             "Number of reduced-basis points.")
        .def("get_n_reduced_basis_full", &SpaceGroup::get_n_reduced_basis_full,
             "Number of irreducible-basis points (full space group).")
        .def("get_n_symmetry_ops", &SpaceGroup::get_n_symmetry_ops,
             "Number of symmetry operations.")
        .def("get_reduced_basis_indices", &SpaceGroup::get_reduced_basis_indices,
             "Flat indices of reduced-basis points in full grid.")
        .def("get_full_to_reduced_map", &SpaceGroup::get_full_to_reduced_map,
             "Map from full grid index to reduced-basis index.")
        .def("get_orbit_counts", &SpaceGroup::get_orbit_counts,
             "Number of equivalent points for each reduced-basis point.")
        .def("physical_basis_name", &SpaceGroup::physical_basis_name,
             "Name of the active physical basis: 'irreducible', 'pmmm', 'm3' or 'z-mirror'.")
        .def("display_info", &SpaceGroup::display_info,
             "Print the group, its crystal system and the sizes of its bases.")
        .def("in_use", &SpaceGroup::in_use,
             "True once a computation box or solver holds this group; the physical basis is then frozen.")
        .def("enable_best_physical_basis", [](SpaceGroup& sg)
        {
            sg.enable_best_physical_basis();
            return sg.physical_basis_name();
        }, "Enable the physical basis that SCFT uses (m3, Pmmm or z-mirror, chosen from the angles the group fixes; "
           "see docs/theory/SpaceGroup.md); the irreducible basis stays when none applies. Returns physical_basis_name().")
        .def("enable_pmmm_physical_basis", &SpaceGroup::enable_pmmm_physical_basis,
             "Enable Pmmm physical basis (1/8 grid) mapping.")
        .def("m3_crysfft_feasible", &SpaceGroup::m3_crysfft_feasible,
             "True when the recursive 3m CrysFFT pays off on this grid (even sizes, Nz/2 >= 8, translations that "
             "fold onto the grid): the rule by which SCFT chooses the m3 basis. An m3 basis enabled on a smaller "
             "grid (Nz/2 >= 2) runs with the recursive 3m CrysFFT all the same.")
        .def("enable_m3_physical_basis", &SpaceGroup::enable_m3_physical_basis,
             "Enable M3 physical basis (1/8 even-index grid) mapping.")
        .def("enable_z_mirror_physical_basis", &SpaceGroup::enable_z_mirror_physical_basis,
             "Enable z-mirror physical basis (half grid along z).")
        .def("to_reduced_basis", [](SpaceGroup& obj, real_in full_field)
        {
            py::buffer_info buf = full_field.request();
            const int n_fields = sg_field_count(buf, obj.get_total_grid(), "total_grid");
            py::array_t<double> out = sg_output_like(buf, n_fields, obj.get_n_reduced_basis());
            obj.to_reduced_basis(static_cast<double*>(buf.ptr), static_cast<double*>(out.request().ptr), n_fields);
            return out;
        }, "Full-grid field(s) -> reduced basis. A 1-D input gives a 1-D output, an (n_fields, total_grid) input an (n_fields, n_reduced) output.")
        .def("reduced_to_irreducible", [](SpaceGroup& obj, real_in reduced_field)
        {
            py::buffer_info buf = reduced_field.request();
            const int n_fields = sg_field_count(buf, obj.get_n_reduced_basis(), "n_reduced_basis");
            py::array_t<double> out = sg_output_like(buf, n_fields, obj.get_n_reduced_basis_full());
            obj.reduced_to_irreducible(static_cast<double*>(buf.ptr), static_cast<double*>(out.request().ptr), n_fields);
            return out;
        }, "Reduced-basis field(s) -> irreducible basis (one value per full-group orbit). A 1-D input gives a 1-D output, an (n_fields, n_reduced) input an (n_fields, n_irreducible) output.")
        .def("irreducible_to_reduced", [](SpaceGroup& obj, real_in irreducible_field)
        {
            py::buffer_info buf = irreducible_field.request();
            const int n_fields = sg_field_count(buf, obj.get_n_reduced_basis_full(), "n_irreducible_basis");
            py::array_t<double> out = sg_output_like(buf, n_fields, obj.get_n_reduced_basis());
            obj.irreducible_to_reduced(static_cast<double*>(buf.ptr), static_cast<double*>(out.request().ptr), n_fields);
            return out;
        }, "Irreducible-basis field(s) -> reduced (physical) basis. A 1-D input gives a 1-D output, an (n_fields, n_irreducible) input an (n_fields, n_reduced) output.")
        .def("from_reduced_basis", [](SpaceGroup& obj, real_in reduced_field)
        {
            py::buffer_info buf = reduced_field.request();
            const int n_fields = sg_field_count(buf, obj.get_n_reduced_basis(), "n_reduced_basis");
            py::array_t<double> out = sg_output_like(buf, n_fields, obj.get_total_grid());
            obj.from_reduced_basis(static_cast<double*>(buf.ptr), static_cast<double*>(out.request().ptr), n_fields);
            return out;
        }, "Reduced-basis field(s) -> full grid. A 1-D input gives a 1-D output, an (n_fields, n_reduced) input an (n_fields, total_grid) output.")
        .def("symmetrize", [](SpaceGroup& obj, real_in field)
        {
            py::buffer_info buf = field.request();
            const int n_fields = sg_field_count(buf, obj.get_total_grid(), "total_grid");
            py::array_t<double> result(buf.shape);   // same shape as the input
            obj.symmetrize(static_cast<double*>(buf.ptr), static_cast<double*>(result.request().ptr), n_fields);
            return result;
        }, "Average field(s) over the space-group orbits. Returns an array of the same shape as the input.");
}

/**
 * @brief Main pybind11 module definition.
 *
 * Creates the `_core` Python module containing all bindings.
 *
 * **Module Contents:**
 *
 * Classes (non-templated):
 * - Polymer: Single polymer chain with topology
 * - Molecules: Container for all polymers and solvents
 * - PropagatorComputationOptimizer: Optimized propagator scheduling
 * - SpaceGroup: Space-group symmetry and reduced bases
 * - PlatformSelector: Platform detection and factory creation
 *
 * Template classes (Real and Complex variants):
 * - ComputationBoxReal, ComputationBoxComplex
 * - PropagatorComputation_Real, PropagatorComputation_Complex
 * - AndersonMixing_Real, AndersonMixing_Complex
 * - FourierTransform_Real, FourierTransform_Complex (with FourierMultiplier_*)
 * - AbstractFactory_Real, AbstractFactory_Complex
 *
 * **Platform Selection:**
 *
 * Use PlatformSelector.create_factory() to get the appropriate factory:
 * @code{.py}
 * factory = _core.PlatformSelector.create_factory("cuda", type="real")
 * factory = _core.PlatformSelector.create_factory("cpu-fftw", type="complex")
 * @endcode
 *
 * The factory then creates all necessary platform-specific objects.
 */
PYBIND11_MODULE(_core, m)
{
    m.attr("build_info") = polymerfts_build_info();
    m.def("numerical_methods", []() {
        py::dict result;
        for (const auto& info : numerical_methods)
        {
            py::dict entry;
            entry["chain_model"] = info.chain_model;
            entry["solver_type"] = info.solver_type;
            entry["contour_order"] = info.contour_order;
            entry["supports_complex"] = info.supports_complex;
            entry["requires_orthogonal_cell"] = info.requires_orthogonal_cell;
            entry["supports_mixed_boundaries"] = info.supports_mixed_boundaries;
            entry["supports_stress"] = info.supports_stress;
            py::list backends;
#ifdef USE_CPU_MKL
            backends.append("cpu-mkl");
#endif
#ifdef USE_CPU_FFTW
            backends.append("cpu-fftw");
#endif
#ifdef USE_CUDA
            backends.append("cuda");
#endif
            entry["compiled_backends"] = backends;
            result[py::str(info.name)] = entry;
        }
        return result;
    }, "Registered propagator methods and their compiled capabilities.");
    m.def("resolve_numerical_method", [](const std::string& chain_model, const std::string& numerical_method) {
        return std::string(resolve_numerical_method(chain_model, numerical_method).name);
    }, py::arg("chain_model"), py::arg("numerical_method") = "",
       "The method the factories use; an empty name gives rqm4 (continuous) or ps (discrete).");
    m.def("runtime_info", [](bool include_cuda) {
        py::dict info;
#ifdef USE_CPU_FFTW
        info["fftw"] = std::string(fftw_version);
#endif
#ifdef USE_CPU_MKL
        char mkl_version[256] = {};
        MKL_Get_Version_String(mkl_version, sizeof(mkl_version));
        info["mkl"] = std::string(mkl_version);
#endif
#ifdef USE_CUDA
        if (include_cuda)
        {
            int version = 0;
            CUDA_CHECK(cudaRuntimeGetVersion(&version));
            info["cuda_runtime"] = version;
            CUDA_CHECK(cudaDriverGetVersion(&version));
            info["cuda_driver_api"] = version;
            CUFFT_CHECK(cufftGetVersion(&version));
            info["cufft"] = version;
            int device = 0;
            cudaDeviceProp properties{};
            CUDA_CHECK(cudaGetDevice(&device));
            CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
            info["cuda_device_name"] = std::string(properties.name);
            info["cuda_compute_capability"] = std::to_string(properties.major) + "." + std::to_string(properties.minor);
            info["cuda_device_memory"] = properties.totalGlobalMem;
            std::string uuid;
            constexpr char hex[] = "0123456789abcdef";
            for (unsigned char byte : properties.uuid.bytes)
            {
                uuid.push_back(hex[byte >> 4]);
                uuid.push_back(hex[byte & 15]);
            }
            info["cuda_device_uuid"] = uuid;
        }
#else
        (void)include_cuda;
#endif
        return info;
    }, py::arg("include_cuda") = false);
    // Grid-size limits of the 32-bit grid indices (GridLimits.h)
    m.attr("MAX_GRID_POINTS") = MAX_GRID_POINTS;
    m.attr("MAX_GRID_ARRAY") = MAX_GRID_ARRAY;
    // Default tolerance of check_total_partition (PropagatorComputation.h)
    m.attr("PARTITION_TOLERANCE") = PARTITION_TOLERANCE;
    // The alias search of the propagator solvers (MinimalFormAlias.h) on a periodic grid of a cell
    // that need not be the simulation box: the smearing in SCFT box relaxation, and WTMD. A box of
    // the cell gives its reciprocal metric, from the angles in degrees as every box.
    struct AliasGrid
    {
        int dim;
        std::array<int, 3> grid;
        IndexMetric metric;
        AliasGrid(const std::vector<int>& nx, const std::vector<double>& lx, const std::vector<double>& angles)
        {
            const ComputationBox<double> box(nx, lx, std::vector<std::string>(2 * nx.size(), "periodic"), angles);
            dim = box.get_dim();
            grid = padded_grid(nx);
            metric = index_metric_of(box.get_recip_metric(), dim);
        }
    };
    m.def("minimal_alias_k_squared", [](const std::vector<int>& nx, const std::vector<double>& lx,
                                        const std::vector<double>& angles)
    {
        const AliasGrid cell(nx, lx, angles);
        py::array_t<double> k_squared(std::vector<py::ssize_t>(nx.begin(), nx.end()));
        alias_k_squared(cell.grid, cell.grid, cell.metric, k_squared.mutable_data());
        return k_squared;
    }, py::arg("nx"), py::arg("lx"), py::arg("angles"),
       "|k|^2 = 4 pi^2 u^T G u of every mode of the periodic grid nx on the cell lx, angles ([alpha, beta, gamma] "
       "or [gamma] in degrees), u the alias of least |k|^2 and G the reciprocal metric of the box; shape nx. "
       "The values of PropagatorSolver.fourier.k_squared() on every mode.");
    m.def("minimal_alias_moments", [](const std::vector<int>& nx, const std::vector<double>& lx,
                                      const std::vector<double>& angles)
    {
        const AliasGrid cell(nx, lx, angles);
        std::vector<py::ssize_t> shape(nx.begin(), nx.end());
        shape.insert(shape.begin(), {cell.dim, cell.dim});
        py::array_t<double> moments(shape);
        alias_moments(cell.grid, cell.grid, cell.metric, cell.dim, moments.mutable_data());
        return moments;
    }, py::arg("nx"), py::arg("lx"), py::arg("angles"),
       "<v_a v_b>, v = G u, over the aliases u tied for the least |k|^2 of every mode of the periodic grid nx on "
       "the cell lx, angles ([alpha, beta, gamma] or [gamma] in degrees), G the reciprocal metric of the box; "
       "shape (dim, dim) + nx. The values of PropagatorSolver.fourier.alias_moments() on every mode.");

    // Bind Block, Polymer, Molecules and the optimizer; SpaceGroup
    bind_molecules(m);
    bind_space_group(m);

    // Bind ComputationBox
    bind_computation_box<double>(m, "Real");
    bind_computation_box<std::complex<double>>(m, "Complex");

    // Bind PropagatorComputation
    bind_propagator_computation<double>(m, "Real");
    bind_propagator_computation<std::complex<double>>(m, "Complex");

    // Bind AndersonMixing
    bind_anderson_mixing<double>(m, "Real");
    bind_anderson_mixing<std::complex<double>>(m, "Complex");

    // Bind FourierTransform
    bind_fourier_transform<double>(m, "Real");
    bind_fourier_transform<std::complex<double>>(m, "Complex");

    // Bind AbstractFactory
    bind_abstract_factory<double>(m, "Real");
    bind_abstract_factory<std::complex<double>>(m, "Complex");

    py::class_<PlatformSelector>(m, "PlatformSelector")
        .def(py::init<>())
        .def_static("avail_platforms", &PlatformSelector::avail_platforms)
        .def_static("create_factory", [](std::string platform_name, bool reduce_memory,
                                         std::string type)
        {
            // Converting type to lowercase
            std::transform(type.begin(), type.end(), type.begin(),
            [](unsigned char c){ return std::tolower(c); });

            // The unique_ptr hands the factory to the Python object
            if (type == "real")
            {
                return py::cast(PlatformSelector::create_factory<double>(platform_name, reduce_memory));
            }
            else if (type == "complex")
            {
                return py::cast(PlatformSelector::create_factory<std::complex<double>>(platform_name, reduce_memory));
            }
            else {
                throw std::runtime_error("Invalid type parameter. Must be either 'real' or 'complex'");
            }
        }, py::arg("platform_name"), py::arg("reduce_memory") = false,
           py::arg("type") = "real");
}
