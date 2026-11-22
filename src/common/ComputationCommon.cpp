/**
 * @file ComputationCommon.cpp
 * @brief Platform-neutral base of the propagator computation classes.
 */

#include <cmath>
#include <algorithm>
#include <memory>
#include <complex>
#include <type_traits>

#include <omp.h>

#include "ComputationCommon.h"
#include "JunctionProduct.h"
#include "ValidationUtils.h"

namespace
{
/// Error text for a junction that would raise complex propagators to a power that is not an
/// integer of at most 10^6 (junction_product::integer_power). The power is that of one branch:
/// the multiplicity of its block, or one less for the propagator that enters a copy, summed
/// over identical branches.
std::string complex_real_power_message(double power)
{
    return "A junction raises complex propagators to the power " + validation::shortest_text(power)
        + ". Complex fields take only integer powers up to 10^6: a non-integer power, which comes from "
        "a non-integer block multiplicity, has no unique value for complex numbers. With complex fields "
        "use integer multiplicities and at most 10^6 identical branches at one junction.";
}
}  // namespace

template <typename T>
ComputationCommon<T>::ComputationCommon(
    ComputationBox<T>* cb,
    Molecules* molecules,
    PropagatorComputationOptimizer* propagator_computation_optimizer,
    const NumericalMethodInfo& numerical_method,
    const std::function<std::unique_ptr<ComputationBackend<T>>()>& create_backend)
    : PropagatorComputation<T>(cb, molecules, propagator_computation_optimizer),
      numerical_method(numerical_method), memory(backend), n_streams(0)
{
    if (propagator_computation_optimizer->get_computation_propagators().size() == 0)
        throw_with_line_number("There is no propagator code. Add polymers first.");
    if (propagator_computation_optimizer->get_computation_blocks().size() == 0)
        throw_with_line_number("There is no block. Add polymers first.");

    // Complex fields take only integer powers at the junctions of branches, checked
    // once here: the backends multiply complex junction products without a check
    if constexpr (!std::is_same_v<T, double>)
    {
        for (const auto& [key, edge] : propagator_computation_optimizer->get_computation_propagators())
        {
            if (edge.start != PropagatorStart::JUNCTION)
                continue;
            for (const auto& dep : edge.deps)
                if (!junction_product::integer_power(std::get<2>(dep)))
                    throw_with_line_number(complex_real_power_message(std::get<2>(dep)));
        }
    }

    // The factory's SpaceGroupGuard attached the group of this computation to the box.
    this->space_group_ = cb->get_space_group();

    backend = create_backend();
    n_streams = backend->get_n_streams();

    for (const auto& item : propagator_computation_optimizer->get_computation_blocks())
        phi_block[item.first] = memory.allocate();

    phi_block_iterators.reserve(phi_block.size());
    for (auto it = phi_block.cbegin(); it != phi_block.cend(); ++it)
        phi_block_iterators.push_back(it);

    for (int s = 0; s < molecules->get_n_solvent_types(); s++)
        phi_solvent.push_back(memory.allocate());
}

template <typename T>
void ComputationCommon<T>::set_fields(
    const std::map<std::string, const T*>& w_input,
    const std::map<std::string, const T*>& q_init)
{
    this->check_box_space_group();
    this->clear_computed();
    for (const auto& [key, edge] : this->propagator_computation_optimizer->get_computation_propagators())
    {
        if (!w_input.contains(edge.monomer_type))
            throw_with_line_number("monomer_type \"" + edge.monomer_type + "\" is not in w_input.");
        if (edge.start == PropagatorStart::GRAFTED && !q_init.contains(edge.graft_label))
            throw_with_line_number("Could not find q_init[\"" + edge.graft_label + "\"]. Pass q_init to run() for grafted polymers.");
    }

    // w keeps the fields of earlier calls; only the types of this call are current
    w_input_types.clear();
    std::map<std::string, BackendArray<T>*> w_current;
    for (const auto& [monomer_type, w_monomer] : w_input)
    {
        if (!w.contains(monomer_type))
            w[monomer_type] = memory.allocate();
        backend->copy_from_host(0, w[monomer_type], w_monomer);
        w_input_types.insert(monomer_type);
        w_current[monomer_type] = w[monomer_type];
    }

    backend->update_dw(w_current);
}

template <typename T>
void ComputationCommon<T>::check_real_power_partitions() const
{
    for (int p = 0; p < this->molecules->get_n_polymer_types(); p++)
    {
        const T q = this->single_polymer_partitions[p];
        if (!std::isnan(std::real(q)) && !std::isnan(std::imag(q)))
            continue;
        for (const Block& block : this->molecules->get_polymer(p).get_blocks())
        {
            if (junction_product::integer_power(block.multiplicity))
                continue;
            throw_with_line_number("The partition function of polymer " + std::to_string(p) + " is NaN. "
                "If the fields are finite, the cause is its block multiplicity of "
                + validation::shortest_text(block.multiplicity) + ": it raises propagators to a "
                "non-integer power, which is undefined where a numerical propagator is negative (for "
                "example with Crank-Nicolson next to absorbing walls, or with a pseudo-spectral method "
                "in a strong field). Use an integer multiplicity, or a method and resolution that keep "
                "the propagators positive.");
        }
    }
}

template <typename T>
void ComputationCommon<T>::compute_solvent_concentrations()
{
    const double ds = this->molecules->get_contour_length_mapping().get_global_ds();
    for (int s = 0; s < this->molecules->get_n_solvent_types(); s++)
    {
        const double volume_fraction   = std::get<0>(this->molecules->get_solvent(s));
        const std::string monomer_type = std::get<1>(this->molecules->get_solvent(s));
        if (!w_input_types.contains(monomer_type))
            throw_with_line_number("monomer_type \"" + monomer_type + "\" of solvent " + std::to_string(s) + " is not in w_input.");

        this->single_solvent_partitions[s] = backend->solvent_concentration(
            phi_solvent[s], w[monomer_type], ds, volume_fraction);
    }
}

template <typename T>
double ComputationCommon<T>::contour_step(const std::string& key) const
{
    const int ds_index = this->propagator_computation_optimizer->get_computation_propagator(key).ds_index;
    return this->molecules->get_contour_length_mapping().get_ds_from_index(ds_index);
}

template <typename T>
T ComputationCommon<T>::concentration_norm(const std::tuple<int, std::string, std::string>& block_key) const
{
    const int p = std::get<0>(block_key);
    const auto& block = this->propagator_computation_optimizer->get_computation_block(block_key);
    Polymer& pc = this->molecules->get_polymer(p);
    return (contour_step(std::get<2>(block_key))*pc.get_volume_fraction()/pc.get_alpha()*block.n_repeated)
           /this->single_polymer_partitions[p];
}

template <typename T>
void ComputationCommon<T>::run_concurrently(size_t n, const std::function<void(int, size_t)>& body)
{
    ParallelErrors errors;
    #pragma omp parallel for num_threads(n_streams)
    for (size_t i = 0; i < n; i++)
        errors.run([&] { body(omp_get_thread_num(), i); });
    backend->synchronize();
    errors.rethrow();
}

template <typename T>
std::vector<typename ComputationCommon<T>::BlockMap::const_iterator> ComputationCommon<T>::first_block_of_each_polymer() const
{
    // phi_block is sorted by polymer
    std::vector<typename BlockMap::const_iterator> first_blocks;
    int current_p = 0;
    for (auto it = phi_block.cbegin(); it != phi_block.cend(); ++it)
    {
        if (std::get<0>(it->first) != current_p)
            continue;
        first_blocks.push_back(it);
        current_p++;
    }
    return first_blocks;
}

template <typename T>
const std::string& ComputationCommon<T>::chain_propagator_key(int polymer, int v, int u, int n, int first_segment)
{
    this->check_box_space_group();
    this->require_propagators();
    // For debugging and testing: the propagator of one block and direction
    const std::string& key = this->propagator_computation_optimizer->get_propagator_key(polymer, v, u);

    if (!this->propagator_computation_optimizer->get_computation_propagators().contains(key))
        throw_with_line_number("Could not find the propagator code '" + key + "'. Disable 'aggregation' option to obtain propagator_computation_optimizer.");

    const int N_RIGHT = this->propagator_computation_optimizer->get_computation_propagator(key).max_n_segment;
    if (n < first_segment || n > N_RIGHT)
        throw_with_line_number("n (" + std::to_string(n) + ") must be in range [" + std::to_string(first_segment)
            + ", " + std::to_string(N_RIGHT) + "]");
    return key;
}

template <typename T>
void ComputationCommon<T>::add_partition_sample(
    ContourPartitions& partitions, int p, const ComputationBlock& block,
    BackendArray<T>* q, BackendArray<T>* q_dagger, BackendArray<T>* bead_factor)
{
    const double n_propagators = block.n_copies;
    T total_partition = (bead_factor != nullptr ? backend->inner_product_inverse_weight(q, q_dagger, bead_factor)
                                                : backend->inner_product(q, q_dagger))
                        *(block.n_repeated/this->cb->get_volume());
    total_partition /= n_propagators;
    partitions.values[p].push_back(total_partition);
    if (this->molecules->get_polymer(p).is_absent())
    {
        double integral = backend->absolute_inner_product(q, q_dagger, bead_factor)
                          *(block.n_repeated/this->cb->get_volume());
        integral /= n_propagators;
        partitions.absolute_integrals[p].push_back(integral);
    }
}

template <typename T>
std::vector<double> ComputationCommon<T>::partition_errors(const ContourPartitions& partitions) const
{
    std::vector<double> errors;
    errors.reserve(this->single_polymer_partitions.size());
    for (size_t p = 0; p < this->single_polymer_partitions.size(); ++p)
    {
        if (this->molecules->get_polymer(static_cast<int>(p)).is_absent())
            errors.push_back(absolute_integrand_partition_error(partitions.values[p], this->single_polymer_partitions[p],
                                                               partitions.absolute_integrals[p]));
        else
            errors.push_back(relative_partition_error(partitions.values[p], this->single_polymer_partitions[p]));
    }
    return errors;
}

template <typename T>
void ComputationCommon<T>::scale_by_contour_step(std::array<T, 6>& stress, const std::string& key_right) const
{
    const double ds = contour_step(key_right);
    for (int d = 0; d < 6; d++)
        stress[d] *= ds;
}

template <typename T>
void ComputationCommon<T>::set_stress_from_blocks(const std::vector<std::array<T, 6>>& stress_of_block)
{
    for (int p = 0; p < this->molecules->get_n_polymer_types(); p++)
        for (int d = 0; d < 6; d++)
            this->dq_dl[p][d] = 0.0;
    size_t b = 0;
    for (const auto& item : this->phi_block)
    {
        for (int d = 0; d < 6; d++)
            this->dq_dl[std::get<0>(item.first)][d] += stress_of_block[b][d];
        b++;
    }
    convert_stress_to_lattice_derivatives();
}

template <typename T>
void ComputationCommon<T>::check_stress_supported() const
{
    if (!numerical_method.supports_stress)
        throw_with_line_number("The " + std::string(numerical_method.name) + " method does not support stress computation.");
}

template <typename T>
void ComputationCommon<T>::convert_stress_to_lattice_derivatives()
{
    // The solvers sum v⊗v components directly, where v = 2π g⁻¹ m is the
    // deformation vector (units: 1/L²): V₁₁ = Σ(kernel × v₁²), V₁₂ = Σ(kernel × v₁v₂), ...
    //
    // The metric tensor g = hᵀh has simple derivatives:
    //   g₁₁ = L₁², g₂₂ = L₂², g₃₃ = L₃²
    //   g₁₂ = L₁L₂cosγ, g₁₃ = L₁L₃cosβ, g₂₃ = L₂L₃cosα
    //
    // Lattice parameter derivatives:
    //   ∂H/∂L₁ ∝ L₁V₁₁ + L₂cosγ·V₁₂ + L₃cosβ·V₁₃
    //   ∂H/∂γ  ∝ -L₁L₂sinγ·V₁₂
    //
    // @see docs/theory/StressTensor.md for derivation
    const int DIM = this->cb->get_dim();
    const int M   = this->cb->get_total_grid();

    const double L1 = this->cb->get_lx(0);
    const double L2 = (DIM >= 2) ? this->cb->get_lx(1) : 1.0;
    const double L3 = (DIM >= 3) ? this->cb->get_lx(2) : 1.0;

    const std::vector<double> angles = this->cb->get_angles();
    const double cos_a = std::cos(angles[0]);  // alpha: between b and c
    const double cos_b = std::cos(angles[1]);  // beta: between a and c
    const double cos_g = std::cos(angles[2]);  // gamma: between a and b
    const double sin_a = std::sin(angles[0]);
    const double sin_b = std::sin(angles[1]);
    const double sin_g = std::sin(angles[2]);

    // Normalization factor (from the Boltzmann factor derivative); local_ds is
    // already multiplied per block by the caller.
    const double norm = -3.0 * M * M;

    for (int p = 0; p < this->molecules->get_n_polymer_types(); p++)
    {
        T V_11 = this->dq_dl[p][0];
        T V_22 = (DIM >= 2) ? this->dq_dl[p][1] : T(0.0);
        T V_33 = T(0.0), V_12 = T(0.0), V_13 = T(0.0), V_23 = T(0.0);
        if (DIM == 3)
        {
            V_33 = this->dq_dl[p][2];
            V_12 = this->dq_dl[p][3];
            V_13 = this->dq_dl[p][4];
            V_23 = this->dq_dl[p][5];
        }
        else if (DIM == 2)
        {
            V_12 = this->dq_dl[p][2];
        }

        // ∂H/∂Lᵢ = Σⱼₖ Vⱼₖ ∂gⱼₖ/∂Lᵢ (the factor 2 of the diagonal is in the normalization)
        this->dq_dl[p][0] = (L1*V_11 + L2*cos_g*V_12 + L3*cos_b*V_13) / norm;
        if (DIM >= 2)
            this->dq_dl[p][1] = (L2*V_22 + L1*cos_g*V_12 + L3*cos_a*V_23) / norm;
        if (DIM >= 3)
            this->dq_dl[p][2] = (L3*V_33 + L1*cos_b*V_13 + L2*cos_a*V_23) / norm;

        // ∂g₁₂/∂γ = -L₁L₂sinγ → ∂H/∂γ = -L₁L₂sinγ·V₁₂, and likewise for β and α
        if (DIM == 3)
        {
            this->dq_dl[p][3] = -L1 * L2 * sin_g * V_12 / norm;  // dH/dγ
            this->dq_dl[p][4] = -L1 * L3 * sin_b * V_13 / norm;  // dH/dβ
            this->dq_dl[p][5] = -L2 * L3 * sin_a * V_23 / norm;  // dH/dα
        }
        else if (DIM == 2)
        {
            this->dq_dl[p][2] = -L1 * L2 * sin_g * V_12 / norm;  // dH/dγ
        }
    }
}

template <typename T>
void ComputationCommon<T>::update_laplacian_operator()
{
    this->check_box_space_group();
    this->clear_computed();
    try
    {
        backend->update_laplacian_operator();
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
void ComputationCommon<T>::set_force_off_diagonal_stress(bool force)
{
    backend->set_force_off_diagonal_stress(force);
}

template <typename T>
void ComputationCommon<T>::advance_propagator_single_segment(T* q_init, T* q_out, int p, int v, int u)
{
    this->check_box_space_group();
    // The step uses the fields and the cell of the last compute_propagators()
    this->require_propagators();
    const Block& block = this->molecules->get_polymer(p).get_block(v, u);
    // Discrete chains have one contour step (ds index 0)
    const int ds_index = this->molecules->get_contour_length_mapping().get_ds_index(block.contour_length);

    // Two temporary fields, freed also when a step throws
    auto deallocate = [this](BackendArray<T>* a) { backend->deallocate(a); };
    std::unique_ptr<BackendArray<T>, decltype(deallocate)> q_in(backend->allocate(), deallocate);
    std::unique_ptr<BackendArray<T>, decltype(deallocate)> q_next(backend->allocate(), deallocate);
    backend->copy_from_host(0, q_in.get(), q_init);
    backend->advance_propagator(0, q_in.get(), q_next.get(), block.monomer_type, ds_index);
    backend->synchronize();
    backend->copy_to_host(q_out, q_next.get());
}

template <typename T>
void ComputationCommon<T>::compute_concentrations_and_stress()
{
    // A method without stress computes neither, as in the reduced-memory classes
    check_stress_supported();
    this->compute_concentrations();
    this->compute_stress();
}

template <typename T>
T ComputationCommon<T>::get_total_partition(int polymer)
{
    this->check_box_space_group();
    this->require_propagators();
    validation::require_index(polymer, this->molecules->get_n_polymer_types(), "Index");
    return this->single_polymer_partitions[polymer];
}

template <typename T>
T ComputationCommon<T>::get_solvent_partition(int s)
{
    this->check_box_space_group();
    this->require_concentrations();
    validation::require_index(s, this->molecules->get_n_solvent_types(), "Index");
    return this->single_solvent_partitions[s];
}

template <typename T>
void ComputationCommon<T>::collect_block_concentrations(
    int p, const std::string& monomer_type, std::optional<double> fugacity,
    std::vector<BackendArray<T>*>& arrays, std::vector<T>& weights) const
{
    for (const auto& [key, phi_b] : phi_block)
    {
        const auto& block = this->propagator_computation_optimizer->get_computation_block(key);
        if ((p != -1 && std::get<0>(key) != p) || block.monomer_type != monomer_type || block.n_segment_right == 0)
            continue;
        if (fugacity)
            weights.push_back(*fugacity * contour_step(std::get<2>(key)) * block.n_repeated);
        else if (!this->molecules->get_polymer(std::get<0>(key)).is_absent())
            weights.push_back(concentration_norm(key));
        else
            continue;
        arrays.push_back(phi_b);
    }
}

template <typename T>
void ComputationCommon<T>::get_total_concentration(std::string monomer_type, T* phi)
{
    this->check_box_space_group();
    this->require_concentrations();
    // A name that is not a monomer type (a typo) is an error, not a zero field
    validation::require_string_key(this->molecules->get_bond_lengths(), monomer_type, "the monomer types");
    std::vector<BackendArray<T>*> arrays;
    std::vector<T> weights;
    collect_block_concentrations(-1, monomer_type, std::nullopt, arrays, weights);
    for (int s = 0; s < this->molecules->get_n_solvent_types(); s++)
    {
        if (std::get<1>(this->molecules->get_solvent(s)) == monomer_type)
        {
            arrays.push_back(phi_solvent[s]);
            weights.push_back(1.0);
        }
    }
    backend->sum_to_host(phi, arrays, weights);
}

template <typename T>
void ComputationCommon<T>::get_total_concentration(int p, std::string monomer_type, T* phi)
{
    this->check_box_space_group();
    this->require_concentrations();
    // A name that is not a monomer type (a typo) is an error, not a zero field
    validation::require_string_key(this->molecules->get_bond_lengths(), monomer_type, "the monomer types");
    validation::require_index(p, this->molecules->get_n_polymer_types(), "Index");

    std::vector<BackendArray<T>*> arrays;
    std::vector<T> weights;
    collect_block_concentrations(p, monomer_type, std::nullopt, arrays, weights);
    backend->sum_to_host(phi, arrays, weights);
}

template <typename T>
void ComputationCommon<T>::get_total_concentration_gce(double fugacity, int p, std::string monomer_type, T* phi)
{
    this->check_box_space_group();
    this->require_concentrations();
    // A name that is not a monomer type (a typo) is an error, not a zero field
    validation::require_string_key(this->molecules->get_bond_lengths(), monomer_type, "the monomer types");
    validation::require_non_negative(fugacity, "fugacity");
    validation::require_index(p, this->molecules->get_n_polymer_types(), "Index");

    // An absent GCE species contributes exactly zero even if an unused
    // propagator integral is nonfinite. Validate its arguments first.
    if (fugacity == 0.0)
    {
        std::fill_n(phi, this->cb->get_n_basis(), T(0.0));
        return;
    }

    std::vector<BackendArray<T>*> arrays;
    std::vector<T> weights;
    collect_block_concentrations(p, monomer_type, fugacity, arrays, weights);
    backend->sum_to_host(phi, arrays, weights);
}

template <typename T>
void ComputationCommon<T>::get_block_concentration(int p, T* phi)
{
    this->check_box_space_group();
    this->require_concentrations();
    const int N = this->cb->get_n_basis();
    validation::require_index(p, this->molecules->get_n_polymer_types(), "Index");

    if (this->propagator_computation_optimizer->use_aggregation())
        throw_with_line_number("Disable 'aggregation' option to invoke 'get_block_concentration'.");

    Polymer& pc = this->molecules->get_polymer(p);
    std::vector<Block>& blocks = pc.get_blocks();
    for (size_t b = 0; b < blocks.size(); b++)
    {
        std::string key_left  = this->propagator_computation_optimizer->get_propagator_key(p, blocks[b].v, blocks[b].u);
        std::string key_right = this->propagator_computation_optimizer->get_propagator_key(p, blocks[b].u, blocks[b].v);
        if (key_left < key_right)
            key_left.swap(key_right);

        const auto block_key = std::make_tuple(p, key_left, key_right);
        auto block_it = phi_block.find(block_key);
        if (block_it == phi_block.end())
            throw_with_line_number("Block (" + std::to_string(p) + ", " + key_left + ", " + key_right + ") is not in phi_block.");
        // An absent species has zero density, also when its unused propagators are not finite
        if (pc.is_absent())
        {
            std::fill_n(&phi[b*N], N, T(0.0));
            continue;
        }
        const T norm = concentration_norm(block_key);
        backend->copy_to_host(&phi[b*N], block_it->second);
        for (int i = 0; i < N; i++)
            phi[b*N + i] *= norm;

        // Identical blocks (e.g. the equal arms of a star) share one entry, which
        // holds the copies of all of them; each block gets the share of its copies
        const double n_repeated = this->propagator_computation_optimizer->get_computation_block(block_key).n_repeated;
        const double n_copies = blocks[b].n_copies;
        if (n_repeated != n_copies)
            for (int i = 0; i < N; i++)
                phi[b*N + i] = phi[b*N + i]/n_repeated*n_copies;
    }
}

template <typename T>
void ComputationCommon<T>::get_solvent_concentration(int s, T* phi)
{
    this->check_box_space_group();
    this->require_concentrations();
    validation::require_index(s, this->molecules->get_n_solvent_types(), "Index");
    backend->copy_to_host(phi, phi_solvent[s]);
}

// Explicit template instantiation
#include "TemplateInstantiations.h"
INSTANTIATE_CLASS(ComputationCommon);
