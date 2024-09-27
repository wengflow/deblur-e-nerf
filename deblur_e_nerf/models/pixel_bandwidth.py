import math
import torch
from ..data import datasets
from ..utils import control, modules


class PixelBandwidth(torch.nn.Module):
    """
    This pixel bandwidth model is a 4th-order unity-gain Non-Linear Time
    -Invariant (NLTI) continuous-time system that is formed by a cascade of:
        1. 1 x 2nd-order unity-gain NLTI Low-Pass Filter (LPF), which:
            a. Damping Ratio
               `zeta = (tau_in + tau_out + (A_amp + 1) * tau_mil) /
                       (2 * sqrt((tau_in + tau_mil) * tau_out * (A_loop + 1)))`
            b. Natural (Angular) Frequency
               `omega_n = sqrt((A_loop + 1) / ((tau_in + tau_mil) * tau_out))`
           It models the dynamic/transient response of the photoreceptor
           circuit.
        2. 2 x 1st-order unity-gain Linear Time-Invariant (LTI) LPFs, which
           cutoff frequencies are, respectively, given by:
            a. `f_c_sf = omega_c_sf / (2 * pi) = 1 / (2 * pi * tau_sf)`
            b. `f_c_diff = omega_c_diff / (2 * pi) = 1 / (2 * pi * tau_diff)`
           They, respectively, model the dynamic/transient response of the:
            a. source follower buffer
            b. change/differencing amplifier

    The input to the model is the black level-accounted pixel log-intensity
    `log(it + black_level)` / effective pixel log-intensity `log(it_eff)`,
    which is equal to the photocurrent-equivalent pixel log-intensity
    `log(I / I_p_to_it_ratio)`. The output of the model is the output of the
    differencing amplifier 1st-order LTI LPF.
    
    In practice, the differencing amplifier will be held in reset after an
    event is generated, throughout the refractory period. This effectively
    resets the state (i.e. output) of the differencing amplifier 1st-order LTI
    LPF to its input (i.e. source follower buffer 1st-order LTI LPF output),
    which is considered in this model.
 
    The pixel bandwidth model is implemented as a Linear Time-Varying (LTV)
    discrete-time system by:
        1. Linearizing the NLTI LPF continuous-time sub-system at the steady
           -state, where the sub-system output = sub-system input = next system
           input sample
        2. Discretizing the linearized continuous-time system, assuming
           First-Order Hold (FOH) on the system inputs
    at each instant where the next input sample is available.
  
    NOTE:
        The discretized system is represented as a LTV state-space model in
        non-standard form, where its state match that of its continuous-time
        counterpart, to accommodate variations in input sample time interval
        `sample_dt` and linearization steady-states (for NLTI sub-system)
    """
    TAU_IN_IT_EFF_PROD_KEY = "input_time_const_eff_it_prod"
    TAU_MIL_IT_EFF_PROD_KEY = "miller_time_const_eff_it_prod"
    A_AMP_KEY = "amplifier_gain"
    A_CL_KEY = "closed_loop_gain"
    TAU_OUT_KEY = "output_time_const"
    F_C_SF_KEY = "sf_cutoff_freq"
    F_C_DIFF_KEY = "diff_amp_cutoff_freq"
    NS_TO_S = 1e-9

    def __init__(
        self,
        dataset_directory,
        min_ts,                                                                 # in nanoseconds
        f_c_dominant_min,                                                       # in Hz
        target_cumprob
    ):
        super().__init__()

        # save some hyperparameters as attributes
        self.omega_c_dominant_min = 2 * math.pi * f_c_dominant_min              # in rad/s

        # define min. ts. & target proposal dist. cumulative prob. buffers
        if torch.is_tensor(min_ts):
            min_ts = min_ts.detach().clone()
        else:
            min_ts = torch.tensor(min_ts)
        self.register_buffer("min_ts", min_ts, persistent=False)                # in nanoseconds
        self.register_buffer("target_cumprob_max_sample_lifetime",
                             torch.tensor(target_cumprob.max_sample_lifetime),
                             persistent=False)

        # extract calibrated pixel bandwidth model parameters
        # (represented in SI units, where applicable)
        camera_calibration = datasets.Event.load_camera_calibration(
            dataset_directory
        )

        calibrated_tau_in_it_eff_prod = torch.from_numpy(
            camera_calibration[self.TAU_IN_IT_EFF_PROD_KEY]
        )
        calibrated_tau_mil_it_eff_prod = torch.from_numpy(
            camera_calibration[self.TAU_MIL_IT_EFF_PROD_KEY]
        )
        calibrated_A_amp = torch.from_numpy(
            camera_calibration[self.A_AMP_KEY]
        )
        calibrated_A_cl = torch.from_numpy(
            camera_calibration[self.A_CL_KEY]
        )
        calibrated_tau_out = torch.from_numpy(
            camera_calibration[self.TAU_OUT_KEY]
        )
        calibrated_f_c_sf = torch.from_numpy(
            camera_calibration[self.F_C_SF_KEY]
        )
        calibrated_f_c_diff = torch.from_numpy(
            camera_calibration[self.F_C_DIFF_KEY]
        )

        # define module buffer/parameters from calibrated values
        calibrated_A_amp_inv = 1 / calibrated_A_amp
        calibrated_A_loop_inv = calibrated_A_cl / calibrated_A_amp
        calibrated_tau_sf = 1 / (2 * math.pi * calibrated_f_c_sf)
        calibrated_tau_diff = 1 / (2 * math.pi * calibrated_f_c_diff)

        """
        NOTE:
            1. Since `tau_in = tau_in_it_eff_prod / it_eff`, `tau_in` is
               invariant to the common scale of `tau_in_it_eff_prod` & NeRF
               -rendered (effective) intensity `it_eff` (Similar for `tau_mil`)
            2. If the pixel bandwidth model parameters are unknown, defining
               `tau_in_it_eff_prod` as a constant & `tau_mil_it_eff_prod` as a
               param. yields a minimal parameterization of `tau_in` & `tau_mil`
            3. With `tau_in_it_eff_prod` being a constant, in theory, the pixel
               bandwidth model implicitly enforces a scale (& gamma)-accurate
               NeRF-rendered (effective) intensity `it_eff`, which
               interpretation is defined by the given `tau_in_it_eff_prod`
            4. Since the NeRF-rendered (effective) intensity `it_eff` is lower
               bounded by `min_modeled_intensity`, the given
               `tau_in_it_eff_prod` constant should be sufficiently large
        """
        self.register_buffer("tau_in_it_eff_prod",
                             calibrated_tau_in_it_eff_prod, persistent=False)
        self.tau_mil_it_eff_prod = torch.nn.parameter.Parameter(
            calibrated_tau_mil_it_eff_prod
        )
        self.A_amp_inv = torch.nn.parameter.Parameter(calibrated_A_amp_inv)
        self.A_loop_inv = torch.nn.parameter.Parameter(calibrated_A_loop_inv)
        self.tau_out = torch.nn.parameter.Parameter(calibrated_tau_out)
        self.tau_sf = torch.nn.parameter.Parameter(calibrated_tau_sf)
        self.tau_diff = torch.nn.parameter.Parameter(calibrated_tau_diff)

        # parameterize pixel bandwidth model parameters via a softplus function
        # s.t. their value are always positive
        softplus = modules.Softplus(beta=1)
        for param_name in ( "tau_mil_it_eff_prod", "A_amp_inv", "A_loop_inv",
                            "tau_out", "tau_sf", "tau_diff" ):
            torch.nn.utils.parametrize.register_parametrization(
                self, param_name, softplus
            )

        # cache linearized system C & D matrices as buffers
        self.register_buffer("linearized_sys_C",                                # (2, 4)
                             torch.tensor([[0, 0, 1, 0],
                                           [0, 0, 0, 1]],
                                          dtype=torch.get_default_dtype()),
                             persistent=False)
        self.register_buffer("linearized_sys_D",                                # (2, 1)
                             torch.zeros(2, 1),
                             persistent=False)
        
    @property
    def A_amp(self):
        return 1 / self.A_amp_inv

    @property
    def A_loop(self):
        return 1 / self.A_loop_inv
    
    @property
    def omega_c_sf(self):
        return 1 / self.tau_sf
    
    @property
    def omega_c_diff(self):
        return 1 / self.tau_diff
        
    def linearized_sys_params(self, steady_state_intensity):                    # (...)
        # infer linearized 2nd-order NLTI LPF continuous-time sub-system params
        tau_in = self.tau_in_it_eff_prod / steady_state_intensity               # (...)
        tau_mil = self.tau_mil_it_eff_prod / steady_state_intensity             # (...)

        tau_in_plus_tau_mil_mul_tau_out = (tau_in + tau_mil) * self.tau_out     # (...)
        two_zeta_omega_n = (                                                    # (...)
            (tau_in + self.tau_out + (self.A_amp + 1) * tau_mil)
            / tau_in_plus_tau_mil_mul_tau_out
        )
        omega_n_square = (self.A_loop + 1) / tau_in_plus_tau_mil_mul_tau_out    # (...)

        return two_zeta_omega_n, omega_n_square, \
               self.omega_c_sf, self.omega_c_diff

    def linearize_sys(self, steady_state_intensity, output_sf_log_it=False):    # (...)
        # infer linearized 4th-order NLTI LPF continuous-time system parameters
        two_zeta_omega_n, omega_n_square, omega_c_sf, omega_c_diff = (          # (...), (...), (), ()
            self.linearized_sys_params(steady_state_intensity)
        )

        # construct the linearized 4th-order cont-time system state space model
        shape = steady_state_intensity.shape                                    # i.e. (...)
        dtype = steady_state_intensity.dtype
        device = steady_state_intensity.device
        C = self.linearized_sys_C                                               # (2, 4)
        D = self.linearized_sys_D                                               # (2, 1)
        if not output_sf_log_it:
            C = C[1, :].unsqueeze(dim=0)                                        # (1, 4)
            D = D[1, :].unsqueeze(dim=0)                                        # (1, 1)
        linearized_sys = control.StateSpace(
            A=torch.zeros(*shape, 4, 4, dtype=dtype, device=device),            # (..., 4, 4)
            B=torch.zeros(*shape, 4, 1, dtype=dtype, device=device),            # (..., 4, 1)
            C=C.expand(*shape, -1, -1),                                         # (..., 1/2, 4)
            D=D.expand(*shape, -1, -1)                                          # (..., 1/2, 1)
        )

        linearized_sys.A[..., 0, 0] = -two_zeta_omega_n
        linearized_sys.A[..., 0, 1] = -omega_n_square
        linearized_sys.A[..., 1, 0] =  1
        linearized_sys.A[..., 2, 1] =  omega_c_sf
        linearized_sys.A[..., 2, 2] = -omega_c_sf
        linearized_sys.A[..., 3, 2] =  omega_c_diff
        linearized_sys.A[..., 3, 3] = -omega_c_diff

        linearized_sys.B[..., 0, 0] =  omega_n_square

        return linearized_sys
    
    def linearized_sys_omega_c_dominant(
        self,
        steady_state_intensity,                                                 # (...)
        reset_diff=False
    ):
        # infer linearized 4th-order NLTI LPF continuous-time system parameters
        two_zeta_omega_n, omega_n_square, omega_c_sf, omega_c_diff = (          # (...), (...), (), ()
            self.linearized_sys_params(steady_state_intensity)
        )

        # deduce the approximate dominant cutoff angular frequency of the
        # linearized 2nd-order NLTI LPF sub-system
        zeta_omega_n = two_zeta_omega_n / 2                                     # (...)
        sigma = zeta_omega_n                                                    # (...)
        j_omega_d = torch.sqrt(zeta_omega_n.square() - omega_n_square)          # (...)

        omega_c_nlti_dominant = sigma - j_omega_d                               # (...)
        omega_n = omega_n_square.sqrt()                                         # (...)
        is_zeta_ge_one = (zeta_omega_n >= omega_n)                              # (...)
        omega_c_nlti_dominant = omega_c_nlti_dominant.where(is_zeta_ge_one,     # (...)
                                                            omega_n)

        # deduce the approximate dominant cutoff angular frequency of the
        # linearized system
        omega_c_dominant = torch.min(omega_c_nlti_dominant, omega_c_sf)         # (...)
        if not reset_diff:
            omega_c_dominant = torch.min(omega_c_dominant, omega_c_diff)        # (...)

        return omega_c_dominant

    @staticmethod
    def discretized_sys_to_weight(discretized_sys):                             # batch of (S-1, ...) `StateSpace` in non-standard form
        """
        Let the (linearized &) discretized system be:
            x[k+1] = A[k] x[k] + B[k] u[k] + B-tilde[k] u[k+1]
              y[k] =    C x[k] +    D u[k]
        Then y[S-1] ≈ \sum_{i=0}^S-1 w[i] x[i] = w[0]x[0] + ... + w[S-1]x[S-1],
            where w[0] = C \varphi(  1, S-1) B[0]
                  w[i] = C \varphi(i+1, S-1) B[i] +
                         C \varphi(  i, S-1) B-tilde[i-1], for i = 1, ..., S-2
                w[S-1] = C B-tilde[S-2] + D
            and \varphi(j, k) = \prod_{i=j}^k-1 A[i] = A[k-1] A[k-2] ... A[j]
        """
        A = discretized_sys.A                                                   # (S-1, ..., n, n)
        B = discretized_sys.B                                                   # (S-1, ..., n, m)
        B_tilde = discretized_sys.B_tilde                                       # (S-1, ..., n, m)
        C = discretized_sys.C[0, ...]                                           # (..., o, n)
        D = discretized_sys.D[0, ...]                                           # (..., o, m)

        # check whether matrices C & D are time-invariant
        assert torch.all(discretized_sys.C == C.unsqueeze(dim=0))
        assert torch.all(discretized_sys.D == D.unsqueeze(dim=0))

        # compute the weights iteratively from sample S-1 to 0
        S = A.shape[0] + 1
        weight = torch.empty(( S, *D.shape ), dtype=D.dtype, device=D.device)   # (S, ..., o, m)

        weight[S-1, ...] = C @ B_tilde[S-2, ...] + D                            # (..., o, m)
        C_varphi_ip1_Sm1 = C                                                    # (..., o, n)
        for i in range(S-2, 0, -1):
            C_varphi_i_Sm1 = C_varphi_ip1_Sm1 @ A[i, ...]                       # (..., o, n)
            weight[i, ...] = C_varphi_ip1_Sm1 @ B[i, ...] \
                             + C_varphi_i_Sm1 @ B_tilde[i-1, ...]               # (..., o, m)
            C_varphi_ip1_Sm1 = C_varphi_i_Sm1
        weight[0, ...] = C_varphi_ip1_Sm1 @ B[0, ...]                           # (..., o, m)

        return weight

    @torch.no_grad()
    def sample_intensity(
        self,
        normalized_interval_gen,                                                # (S-1, ...)
        output_ts,                                                              # (...) in nanoseconds
        intensity_sampling_fn
    ):
        """
        NOTE:
            "Stop-gradient" is applied on the input (effective) pixel intensity
            sample lifetimes `sample_lifetime` to prevent their unintended
            optimization, which may result in undesired effects / subopt. perf.
        """
        S = normalized_interval_gen.shape[0] + 1
        batch_shape = normalized_interval_gen.shape[1:]                         # i.e. ...

        # derive the normalized input (effective) pixel it. sample lifetimes
        normalized_interval_gen_boundary = torch.linspace(                      # (S)
            1, 0, S, dtype=normalized_interval_gen.dtype,
            device=normalized_interval_gen.device
        )
        normalized_interval_gen_boundary = (                                    # (S, 1, ..., 1)
            normalized_interval_gen_boundary.view(
                -1, *(( 1, ) * len(batch_shape))
            )
        )
        normalized_interval_gen = torch.lerp(                                   # (S-1, ...)
            input=normalized_interval_gen_boundary[:-1, ...],
            end=normalized_interval_gen_boundary[1:, ...],
            weight=normalized_interval_gen
        )

        normalized_sample_lifetime = torch.lerp(                                # (S-2, ...)
            input=normalized_interval_gen[:-1, ...],
            end=normalized_interval_gen[1:, ...],
            weight=0.5
        )
        ones = torch.ones_like(                                                 # (1, ...)
            normalized_sample_lifetime[0, ...].unsqueeze(dim=0)
        )
        zeros = torch.zeros_like(ones)                                          # (1, ...)
        normalized_sample_lifetime = torch.cat(                                 # (S, ...)
            ( ones, normalized_sample_lifetime, zeros ), dim=0
        )

        # sample the input (effective) pixel intensity sample lifetimes
        exp_dist = torch.distributions.exponential.Exponential(
            rate=self.NS_TO_S * self.omega_c_dominant_min                       # in rad/ns
        )
        sample_lifetime = exp_dist.icdf(                                        # (S, ...)
            self.target_cumprob_max_sample_lifetime
            * normalized_sample_lifetime
        )

        # derive the sample timestamps & sample the input (effective) pixel it.
        """
        NOTE:
            We assume the input (effective) pixel intensity at timestamps
            smaller than `self.min_ts` is equal to its initial value at
            timestamp `self.min_ts`.
        """
        with torch.enable_grad():
            sample_ts = output_ts - sample_lifetime                             # (S, ...)
            sampling_output = intensity_sampling_fn(
                sample_ts.clamp(min=self.min_ts)
            )
        intensity_sample = sampling_output[0]                                   # (S, ...)
        auxiliary_output = sampling_output[1:]

        return intensity_sample, sample_ts, auxiliary_output
    
    def intensity_sample_to_weight(
        self,
        intensity_sample,                                                       # (S, ...)
        sample_dt,                                                              # (S-1, ...) in nanoseconds
        output_sf_log_it=False
    ):
        assert torch.all(sample_dt > 0)

        # linearize & FOH-discretize the 4th-order NLTI continuous system with
        # the following outputs:
        #   1. differencing amplifier 1st-order LTI LPF output before reset
        #   2. the source follower buffer 1st-order LTI LPF output, if required
        linearization_intensity_sample = intensity_sample[1:, ...]              # (S-1, ...)
        linearized_sys = self.linearize_sys(linearization_intensity_sample,     # batch of (S-1, ...) `StateSpace` in standard form
                                            output_sf_log_it)
        discretized_sys = control.foh_cont2discrete(                            # batch of (S-1, ...) `StateSpace` in non-standard form
            linearized_sys, self.NS_TO_S * sample_dt,
            is_state_preserved=True, is_efficient=True
        )

        # derive & return the unnormalized weights associated to the input
        # (eff.) pixel (log-)it. samples, for synthesizing the differencing
        # amplifier LPF output before reset & possibly also the source follower
        # buffer LPF output, at the timestamp associated to the last sample
        weight = self.discretized_sys_to_weight(discretized_sys)                # (S, ..., 1/2, 1)
        weight = weight.squeeze(dim=-1)                                         # (S, ..., 1/2)

        return weight
    
    def weighted_it_sample_to_output_log_it(
        self,
        weight,                                                                 # (S, ..., 1/2)
        intensity_sample,                                                       # (S, ...)
        last_sample_ts,                                                         # (...) in nanoseconds
        reset_diff=False
    ):
        # normalize the input (effective) pixel (log-)intensity sample weights
        normalized_weight = weight / weight.sum(dim=0, keepdim=True)            # (S, ..., 1/2)

        # synthesize the differencing amplifier LPF output before reset &
        # possibly also the source follower buffer LPF output, at the given
        # timestamp associated to the last input (eff.) pixel (log-)it. sample
        log_intensity_sample = intensity_sample.log()                           # (S, ...)
        log_intensity_sample = log_intensity_sample.unsqueeze(dim=-1)           # (S, ..., 1)
        output_log_intensity = torch.sum(                                       # (..., 1/2)
            normalized_weight * log_intensity_sample, dim=0
        )

        # reset the differencing amplifier LPF, if required, & synthesize its
        # output after reset, at the given last sample timestamp
        if reset_diff:
            sf_log_it = output_log_intensity[..., 0]                            # (...)
            diff_log_it_bfr_reset = output_log_intensity[..., 1]                # (...)
            self.reset_delta_log_it = diff_log_it_bfr_reset - sf_log_it         # (...)
            self.reset_ts = last_sample_ts                                      # (...)

            diff_log_it_aft_reset = sf_log_it                                   # (...)
            """
            NOTE:
                diff_log_it_aft_reset
                = diff_log_it_bfr_reset - self.reset_delta_log_it * torch.exp(
                    -self.omega_c_diff
                     * self.FROM_NANO * (last_sample_ts - self.reset_ts)
                  )
                = sf_log_it
            """
        else:
            diff_log_it_bfr_reset = output_log_intensity[..., 0]                # (...)
            reset_dt = (last_sample_ts - self.reset_ts).to(                     # (...)
                self.omega_c_diff.dtype
            )
            assert torch.all(reset_dt >= 0)

            diff_log_it_aft_reset = (                                           # (...)
                diff_log_it_bfr_reset - self.reset_delta_log_it * torch.exp(
                    -self.omega_c_diff * (self.NS_TO_S * reset_dt)
                )
            )
        output_log_intensity = diff_log_it_aft_reset                            # (...)
        return output_log_intensity 
    
    def forward(
        self,
        normalized_interval_gen,                                                # (S-1, ...)
        output_ts,                                                              # (...) in nanoseconds
        intensity_sampling_fn,
        reset_diff=False
    ):
        """
        Args:
            normalized_interval_gen (torch.Tensor):
                Normalized sample (timestamp / lifetime) interval generator
                samples in the range of [0, 1] with shape (S-1, ...), where S
                is the input (blur-free, effective) pixel intensity sample size
            output_ts (torch.Tensor):
                Timestamp of the output (blurred, effective) pixel
                log-intensity with shape (...) in nanoseconds
            intensity_sampling_fn (Callable):
                Function to sample input (effective) pixel intensities given
                timestamps (in nanoseconds) as inputs. The function should
                return a T-tuple, where the 1st element is the sampled
                intensity in the same shape as the input timestamp, and the
                remaining T-1 elements are auxiliary outputs.
            reset_diff (bool):
                Whether to reset the differencing amplifier 1st-order LTI LPF,
                at the output timestamp
        Returns:
            output_log_intensity (torch.Tensor):
                The pixel bandwidth model / differencing amplifier 1st-order
                LTI LPF output (effective) pixel log-intensity with shape (...)
            auxiliary_output (tuple):
                (T-1)-tuple of auxiliary outputs, where each element is exactly
                the (T-1)-tuple returned by `intensity_sampling_fn()`
        """
        intensity_sample, sample_ts, auxiliary_output = self.sample_intensity(  # (S, ...), (S, ...) in nanoseconds, (T-1)-tuple
            normalized_interval_gen, output_ts, intensity_sampling_fn
        )
        sample_dt = sample_ts.diff(dim=0).to(intensity_sample.dtype)            # (S-1, ...) in nanoseconds
        weight = self.intensity_sample_to_weight(                               # (S, ..., 1/2)
            intensity_sample, sample_dt, output_sf_log_it=reset_diff
        )
        output_log_intensity = self.weighted_it_sample_to_output_log_it(        # (...)
            weight, intensity_sample, output_ts, reset_diff
        )

        return output_log_intensity, auxiliary_output
