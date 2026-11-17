"""Well-Tempered Metadynamics (WTMD) for L-FTS simulations.

This module implements well-tempered metadynamics to enhance sampling
in Langevin field-theoretic simulations of block copolymers.

This implementation follows deep-langevin-fts, except for the half-spectrum weight
of the last kz plane when nz is odd (see WTMD.__init__) and the deposition of the
hills (see WTMD.update_statistics). Ψ and its derivative are computed from scaled
modes, which gives the same values without overflow, and the bias is zero where
the formula of deep-langevin-fts divides zero by zero: at Ψ = 0 and, for ell < 2,
at a zero mode (see WTMD.add_bias).
"""

import numpy as np
import scipy.signal

from ..theory.smearing import periodic_transform
from ..utils.common import atomic_savemat
from ..utils.checkpoint import pair_suffix
from ..utils.validation import ValidationError, _finite_number, right_angle


class WTMD:
    """Well-Tempered Metadynamics for enhanced sampling in L-FTS.

    This implementation matches deep-langevin-fts up to the odd-nz weight and the
    deposition of the hills (see the module docstring). With `fourier`, the PropagatorSolver.fourier of the simulation
    (real fields, periodic box), the order parameter and its derivative are transformed
    on the platform (on the GPU with CUDA) instead of with numpy.fft, in the same rfftn
    layout.
    """

    def __init__(self, nx, lx, nbar,
                 eigenvalues, real_fields_idx,
                 l=4,
                 kc=6.02,
                 dT=5.0,
                 sigma_psi=0.16,
                 psi_min=0.0,
                 psi_max=10.0,
                 dpsi=1e-3,
                 update_freq=1000,
                 recording_period=100000,
                 u=None, up=None, I0=None, I1=None, angles=None, fourier=None):

        _finite_number(l, "wtmd.ell", positive=True)
        if l < 1:
            raise ValidationError("wtmd.ell must be at least 1")
        self.l = l
        self.kc = kc
        self.sigma_psi = sigma_psi
        self.dT = dT
        self.psi_min = psi_min
        self.psi_max = psi_max
        self.dpsi = dpsi
        self.bins = int(np.round((psi_max - psi_min) / dpsi))
        self.update_freq = update_freq
        self.recording_period = recording_period

        if len(nx) != 3:
            raise ValidationError("WTMD requires a 3D grid (nx with three entries)")
        self.nx = nx
        self.lx = lx
        self._fourier = fourier
        self.M = nx[0] * nx[1] * nx[2]
        self.Mk = nx[0] * nx[1] * (nx[2] // 2 + 1)

        # Cell volume and metric (angles [alpha, beta, gamma] in degrees; None: orthogonal)
        cos_a, cos_b, cos_g = np.cos(np.deg2rad(angles)) if angles is not None else (0.0, 0.0, 0.0)
        self.orthogonal = angles is None or all(right_angle(angle) for angle in angles)   # the test of the native box
        if self.orthogonal:
            self.V = lx[0] * lx[1] * lx[2]
        else:
            self.V = lx[0] * lx[1] * lx[2] * np.sqrt(1.0 - cos_a**2 - cos_b**2 - cos_g**2 + 2.0 * cos_a * cos_b * cos_g)
        self.CV = np.sqrt(nbar) * self.V

        # Choose one index of w_aux to use order parameter
        self.eigenvalues = eigenvalues
        self.real_fields_idx = real_fields_idx
        eigen_value_min = 0.0
        self.exchange_idx = None
        for count, i in enumerate(self.real_fields_idx):
            if self.eigenvalues[i] < eigen_value_min:
                eigen_value_min = self.eigenvalues[i]
                self.exchange_idx = i
                self.langevin_idx = count
        if self.exchange_idx is None:
            raise ValidationError("WTMD needs a real auxiliary (exchange) field, i.e. a negative value among the "
                                  "printed 'Eigenvalues' of the auxiliary fields (for two monomer types, a positive chi_n)")

        self.u = np.zeros(self.bins)
        self.up = np.zeros(self.bins)
        self.I0 = np.zeros(self.bins)
        self.I1 = {}

        # Copy data and normalize them by sqrt(nbar)*V
        if u is not None:
            self.u = u.copy() / self.CV
        if up is not None:
            self.up = up.copy() / self.CV
        if I0 is not None:
            self.I0 = I0.copy()
        if I1 is not None:
            self.I1 = {key: np.array(value, dtype=np.float64) / self.CV for key, value in I1.items()}

        # The hills are Gaussians cut at 8σ, where they fall below 1e-13 of their peak. The
        # histogram of the samples extends by this margin (pad bins) on both sides of the
        # range, so that samples just outside the range deposit their tails inside it.
        self.pad = int(np.ceil(8.0 * self.sigma_psi / self.dpsi))
        self.psi_range = np.linspace(self.psi_min, self.psi_max, num=self.bins, endpoint=False)
        self.psi_range_hist = np.linspace(self.psi_min - (self.pad + 0.5) * self.dpsi,
                                          self.psi_max + (self.pad - 0.5) * self.dpsi,
                                          num=self.bins + 2 * self.pad + 1, endpoint=True)

        # Store order parameters for updating U and U_hat
        self.order_parameter_history = []
        # Store dH for updating I1
        self.dH_history = {}

        # Initialize arrays in Fourier spaces
        # Half-spectrum weights: the planes kz = 0 and, for even nz, kz = nz/2 are their own
        # conjugates; every other plane stands for itself and its conjugate
        self.wt = 2 * np.ones([nx[0], nx[1], nx[2] // 2 + 1])
        self.wt[:, :, 0] = 1.0
        if nx[2] % 2 == 0:
            self.wt[:, :, nx[2] // 2] = 1.0

        if self.orthogonal:
            space_kx, space_ky, space_kz = np.meshgrid(
                2 * np.pi / lx[0] * np.concatenate([np.arange((nx[0] + 1) // 2), nx[0] // 2 - np.arange(nx[0] // 2)]),
                2 * np.pi / lx[1] * np.concatenate([np.arange((nx[1] + 1) // 2), nx[1] // 2 - np.arange(nx[1] // 2)]),
                2 * np.pi / lx[2] * np.arange(nx[2] // 2 + 1), indexing='ij')
            mag_k = np.sqrt(space_kx**2 + space_ky**2 + space_kz**2)
        else:
            # |k|^2 = 4π² mᵀ g⁻¹ m with the alias representative that minimizes the form, from the
            # search of the propagator solvers: of the simulation's transform, or of a box of its own
            transform = fourier if fourier is not None else periodic_transform(nx, lx, angles)[0]
            mag_k = np.sqrt(transform.k_squared()[:, :, :nx[2] // 2 + 1])
        self.fk = 1.0 / (1.0 + np.exp(12.0 * (mag_k - kc) / kc))
        self._fk_root = np.power(self.fk, 1.0 / self.l)

        # Gaussian hill and its derivative at the offsets X = (Ψ - Ψ_sample)/σ of the bins
        X = self.dpsi * np.arange(-self.pad, self.pad + 1) / self.sigma_psi
        self.u_kernel = np.exp(-0.5 * X**2)
        self.up_kernel = -X / self.sigma_psi * np.exp(-0.5 * X**2)

    def start_fresh_run(self):
        """Called before a fresh run of the simulation (from step 1): the samples that an earlier run left
        pending are dropped. The bias and statistics stay as the object holds them: those of the
        constructor, of earlier runs or of a checkpoint, or a bias set on the object (an imported bias)."""
        self.order_parameter_history = []
        self.dH_history = {}

    def order_parameter(self, w_aux):
        """The order parameter Ψ of the auxiliary fields w_aux."""
        psi, _, _ = self._order_parameter(w_aux)
        return psi

    def add_bias(self, w_aux, forces):
        """Computes Ψ of the auxiliary fields w_aux, adds the bias V U'(Ψ) dΨ/dw to the forces of the real
        fields (rows in the order of real_fields_idx) and returns Ψ.

        The bias is zero where U'(Ψ) = 0 and at Ψ = 0, which only a zero exchange field has and where
        dΨ/dw (0/0) has no direction.
        """
        psi, w_k, modes = self._order_parameter(w_aux)

        # Calculate current value of U'(Ψ) using linear interpolation
        up_hat = np.interp(psi, self.psi_range, self.up)
        if up_hat == 0.0:
            print("\t[WTMD] Ψ:%8.5f, bias:0 (U'=0)" % psi)
            return psi
        if psi == 0.0:
            dpsi_dwr = np.zeros(self.M)
        else:
            # Derivative of Ψ with respect to the modes w_k
            magnitude = np.absolute(w_k)
            direction = np.divide(w_k, magnitude, out=np.zeros_like(w_k), where=magnitude > 0)
            # f_k^(1/ell) [b_k/||b||_ell]^(ell-1) is bounded and independent
            # of the field scale. Keep the zero subgradient for ell=1 and the
            # zero derivative at a zero coefficient for ell>1.
            dpsi_dwk = self._fk_root * np.power(modes, self.l - 1.0) * direction
            # Derivative of Ψ with respect to the field on the grid
            if self._fourier is not None:
                dpsi_dwr = self._fourier.backward(dpsi_dwk) * self.M / self.V
            else:
                dpsi_dwr = np.fft.irfftn(dpsi_dwk, self.nx, axes=tuple(range(len(self.nx)))) * self.M / self.V

        # Add bias
        bias = np.reshape(self.V * up_hat * dpsi_dwr, self.M)
        forces[self.langevin_idx] += bias

        print("\t[WTMD] Ψ:%8.5f, np.std(dΨ_dwr):%8.5e, np.std(bias):%8.5e" % (psi, np.std(dpsi_dwr), np.std(bias)))
        return psi

    def _order_parameter(self, w_aux):
        """Ψ of w_aux, with the modes w_k of the exchange field (rfftn layout) and the scaled modes
        b_k/||b||_ell that the derivative needs (None at Ψ = 0)."""
        if self._fourier is not None:
            w_k = self._fourier.forward(np.reshape(w_aux[self.exchange_idx], -1))
        else:
            w_k = np.fft.rfftn(np.reshape(w_aux[self.exchange_idx], self.nx))
        # Absorb the cutoff before scaling: b_k = |w_k| f_k^(1/ell).
        # Powers of b/max(b) cannot overflow, even for large ell. The half-
        # spectrum weights are at least one, so b/||b||_ell also stays in [0, 1].
        magnitude = np.absolute(w_k) * self._fk_root
        scale = np.max(magnitude)
        if scale == 0.0:
            return 0.0, w_k, None
        modes = magnitude / scale
        norm = np.power(np.sum(np.power(modes, self.l) * self.wt), 1.0 / self.l)
        modes /= norm
        # norm/M <= 1 for ell >= 1; restore the scale without a large product
        # or an intermediate scale/M that can underflow prematurely.
        return scale * (norm / self.M), w_k, modes

    def store_order_parameter(self, psi, dH):
        self.order_parameter_history.append(psi)
        for key in dH:
            if key not in self.dH_history:
                self.dH_history[key] = []
            self.dH_history[key].append(dH[key])

    def update_statistics(self):

        # Histogram of the samples on the extended bins, divided by the number of all samples:
        # every sample deposits a hill of the same weight, and samples beyond the margin none
        n_samples = len(self.order_parameter_history)
        counts, _ = np.histogram(self.order_parameter_history, bins=self.psi_range_hist)
        hist = counts / (n_samples * self.dpsi)

        # The hills are summed by linear convolution; mode "valid" keeps the bins of the range
        dI1 = {}
        for key in self.dH_history:
            hist_dH_, _ = np.histogram(self.order_parameter_history,
                                       weights=self.dH_history[key],
                                       bins=self.psi_range_hist)
            dI1[key] = scipy.signal.fftconvolve(hist_dH_ / n_samples, self.u_kernel, mode="valid")

        # Compute dU(Ψ), dU'(Ψ)
        amplitude = np.exp(-self.CV * self.u / self.dT) / self.CV
        gaussian = scipy.signal.fftconvolve(hist, self.u_kernel, mode="valid") * self.dpsi
        du = amplitude * gaussian
        dup = amplitude * scipy.signal.fftconvolve(hist, self.up_kernel, mode="valid") * self.dpsi - self.CV * self.up / self.dT * du

        print("np.max(np.abs(amplitude)):%8.5e" % (np.max(np.abs(amplitude))))
        print("np.max(np.abs(gaussian)):%8.5e" % (np.max(np.abs(gaussian))))
        print("np.max(np.abs(du)):%8.5e" % (np.max(np.abs(du))))
        print("np.max(np.abs(dup)):%8.5e" % (np.max(np.abs(dup))))
        for key in dI1:
            print("np.max(np.abs(dI1[%s])):%8.5e" % (key, np.max(np.abs(dI1[key]))))

        # Update u, up, I0, I1
        self.u += du
        self.up += dup
        self.I0 += gaussian
        for key in dI1:
            if key not in self.I1:
                self.I1[key] = np.zeros(self.bins)
            self.I1[key] += dI1[key]

        # Reset lists
        self.order_parameter_history = []
        self.dH_history = {}

    def write_data(self, file_name):
        mdic = {
            **self.settings(),
            "psi_range": self.psi_range,
            "nx": self.nx,
            "lx": self.lx,
            "volume": self.V,
            "nbar": (self.CV / self.V)**2,
            "eigenvalues": self.eigenvalues,
            "real_fields_idx": self.real_fields_idx,
            "exchange_idx": self.exchange_idx,
            "langevin_idx": self.langevin_idx,
            "u": self.u * self.CV,
            "up": self.up * self.CV,
            "I0": self.I0
        }

        # Add I0 and dH_Ψ to the dictionary
        for key in self.I1:
            I1 = self.I1[key] * self.CV
            dH_psi = I1.copy()
            dH_psi[np.abs(self.I0) > 0] /= self.I0[np.abs(self.I0) > 0]
            suffix = pair_suffix(key)
            mdic["I1_" + suffix] = I1
            mdic["dH_psi_" + suffix] = dH_psi

        atomic_savemat(file_name, mdic, long_field_names=True, do_compression=True)

    # ---------------- Checkpoints: settings, state, checks and restore ----------------

    def settings(self):
        """Settings of the bias and its statistics, which define a continuation (checkpoint hash); the
        checkpoints (wtmd_params) and the WTMD files (write_data) store them too."""
        return {key: getattr(self, key) for key in
                ("l", "kc", "dT", "sigma_psi", "psi_min", "psi_max", "dpsi", "bins", "update_freq")}

    def checkpoint_data(self):
        """The bias, the statistics and the pending samples, as a fields file stores them.

        The bias arrays are stored with the factor CV, as in deep-langevin-fts, and without it
        ("wtmd_unscaled_*"), which a continued run restores exactly: (x*CV)/CV can differ from x in
        the last bit. I1 and the samples stored since the last update make a continued run carry the
        same statistics as an uninterrupted one.
        """
        data = {"wtmd_u": self.u * self.CV, "wtmd_up": self.up * self.CV, "wtmd_I0": self.I0,
                "wtmd_unscaled_u": self.u, "wtmd_unscaled_up": self.up}
        for key, value in self.I1.items():
            suffix = pair_suffix(key)
            data["wtmd_I1_" + suffix] = value * self.CV
            data["wtmd_unscaled_I1_" + suffix] = value
        data["wtmd_order_parameter_history"] = np.array(self.order_parameter_history, dtype=np.float64)
        for key, values in self.dH_history.items():
            data["wtmd_dH_history_" + pair_suffix(key)] = np.array(values, dtype=np.float64)
        data["wtmd_params"] = {**self.settings(), "recording_period": self.recording_period}
        return data

    def validate_checkpoint(self, data, pairs):
        """Checks the WTMD state of a checkpoint (checkpoint_data) for the monomer pairs `pairs`
        ("A,B" keys) before a continuation changes any state. A missing key raises KeyError, which
        checkpoint.validate_checkpoint reports as an invalid checkpoint."""
        def check_array(key):
            if np.size(data[key]) != self.bins or not np.all(np.isfinite(data[key])) or np.iscomplexobj(data[key]):
                raise ValidationError(f"Invalid checkpoint WTMD array '{key}'")
        # Newer checkpoints also hold the bias arrays without the factor CV ("wtmd_unscaled_*"), which
        # restore() reads instead of the scaled ones
        unscaled = "wtmd_unscaled_u" in data
        for key in ("wtmd_u", "wtmd_up", "wtmd_I0") + (("wtmd_unscaled_u", "wtmd_unscaled_up") if unscaled else ()):
            check_array(key)
        pending = np.size(data["wtmd_order_parameter_history"])
        if (pending >= self.update_freq or not np.all(np.isfinite(data["wtmd_order_parameter_history"])) or
                np.iscomplexobj(data["wtmd_order_parameter_history"])):
            raise ValidationError("Invalid checkpoint pending WTMD samples")
        for pair in pairs:
            suffix = pair_suffix(pair)
            key = "wtmd_I1_" + suffix
            if np.any(data["wtmd_I0"]) or key in data:
                check_array(key)
                if unscaled:
                    check_array("wtmd_unscaled_I1_" + suffix)
            key = "wtmd_dH_history_" + suffix
            if pending and (np.size(data[key]) != pending or not np.all(np.isfinite(data[key])) or np.iscomplexobj(data[key])):
                raise ValidationError(f"Invalid checkpoint WTMD history '{key}'")

    def restore(self, data, pairs):
        """The bias, the statistics and the pending samples of a checkpoint that validate_checkpoint()
        accepted, for the monomer pairs `pairs` ("A,B" keys). An unversioned checkpoint, which
        validate_checkpoint() does not check, may lack the WTMD state, which then stays as it is, or the
        pending samples. The arrays are restored without the factor CV, or for an older checkpoint from
        the scaled ones divided by CV (as in deep-langevin-fts); I1 is absent before the first update."""
        if "wtmd_u" not in data:
            return
        def bias(name):
            if "wtmd_unscaled_u" in data:
                return np.array(data["wtmd_unscaled_" + name])
            return np.array(data["wtmd_" + name]) / self.CV
        self.u = bias("u")
        self.up = bias("up")
        self.I0 = np.array(data["wtmd_I0"])
        self.I1 = {key: bias("I1_" + pair_suffix(key)) for key in pairs if "wtmd_I1_" + pair_suffix(key) in data}
        self.order_parameter_history = list(np.atleast_1d(data.get("wtmd_order_parameter_history", [])))
        self.dH_history = {}
        if self.order_parameter_history:
            self.dH_history = {key: list(np.atleast_1d(data["wtmd_dH_history_" + pair_suffix(key)])) for key in pairs}
        print("Restored WTMD state")
