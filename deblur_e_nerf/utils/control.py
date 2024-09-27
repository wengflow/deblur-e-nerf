import typing
from dataclasses import dataclass
import torch


@dataclass
class StateSpace:
    """
    (Batched) Continuous-time Linear Time-Invariant (LTI) system state-space
    model in standard form:
            x-dot(t) = A x(t) + B u(t)
                y(t) = C x(t) + D u(t)
    or (Batched) Discrete-time LTI or Linear Time-Varying (LTV) system state
    -space model in:
        1. Standard form
            x[k+1] = A[k] x[k] + B[k] u[k]
              y[k] = C[k] x[k] + D[k] u[k]
        2. Non-standard form
            x[k+1] = A[k] x[k] + B[k] u[k] + B-tilde[k] u[k+1]
              y[k] = C[k] x[k] + D[k] u[k]
    """
    A: torch.Tensor
    B: torch.Tensor
    C: torch.Tensor
    D: torch.Tensor
    B_tilde: typing.Optional[torch.Tensor] = None


def foh_cont2discrete(
    system,
    dt,
    is_state_preserved = False,
    is_efficient=False
):
    """
    Adapted from `scipy.signal.cont2discrete(method='foh')`
    (https://github.com/scipy/scipy/blob/v1.11.3/scipy/signal/_lti_conversion.py#L498-L518)

    Modifications:
    1. Only support state-space representation
    2. Support batched state-space models
    3. Support state preservation after discretization
    4. Support a more computationally & memory efficient implementation
    5. Convert Numpy operations to PyTorch

    Args:
        system (StateSpace): 
            (Batched) Continuous-time, Linear Time-Invariant (LTI) system state
            -space model (in standard form), where:
                1. `system.A` has shape (..., n, n)
                2. `system.B` has shape (..., n, m)
                3. `system.C` has shape (..., o, n)
                4. `system.D` has shape (..., o, m)
        dt (torch.Tensor):
            (Batched) discretization time steps of shape (...)
        is_state_preserved (bool):
            Whether to preserve the state of the continuous LTI system after
            discretization. Specifically, let the continuous LTI system state
            be `x`, input be `u`, and First-Order Hold (FOH)-discretized LTI
            system state be `xi`. If `is_state_preserved` is true, then
            `xi[k] = x[k]`. Else, `xi[k] = x[k] - gamma2 * u[k]`.
        is_efficient (bool):
            Whether to adopt a more computationally & memory efficient
            implementation, which requires `system.A` to be invertible.
    Returns:
        sysd (StateSpace):
            (Batched) Discretized LTI system state-space model. If state is
            preserved, then the state-space model is in standard form. Else, it
            is not.

    References:
        1. G. F. Franklin, J. D. Powell, and M. L. Workman, Digital control of 
           dynamic systems, 3rd ed. Menlo Park, Calif: Addison-Wesley, 
           pp. 204-206, 1998.
    """
    a = system.A                                                                # (..., n, n)
    b = system.B                                                                # (..., n, m)
    c = system.C                                                                # (..., o, n)
    d = system.D                                                                # (..., o, m)

    batch_shape = dt.shape                                                      # ie. (...)
    dt = dt.view(*batch_shape, 1, 1)                                            # (..., 1, 1)

    # Size parameters for convenience
    n, m = b.shape[-2:]

    if is_efficient:
        a_dt = a * dt                                                           # (..., n, n)
        phi = torch.linalg.matrix_exp(a_dt)                                     # (..., n, n)
        a_dt_inv_b_dt = torch.linalg.solve(a, b)                                # (..., n, m)
        gamma1 = (phi - torch.eye(n, dtype=a.dtype, device=a.device)) \
                 @ a_dt_inv_b_dt                                                # (..., n, m)
        gamma2 = torch.linalg.solve(a_dt, gamma1) - a_dt_inv_b_dt               # (..., n, m)
    else:
        # Build an exponential matrix similar to 'zoh' method
        em = torch.zeros(*batch_shape, n + 2*m, n + 2*m,                        # (..., n+2m, n+2m)
                         dtype=a.dtype, device=a.device)
        em[..., :n, :n] = a * dt                                                # (..., n, n)
        em[..., :n, n:n+m] = b * dt                                             # (..., n, m)
        em[..., n:n+m, n+m:] = torch.eye(m, dtype=a.dtype, device=a.device)     # (..., m, m)

        ms = torch.linalg.matrix_exp(em)                                        # (..., n+2m, n+2m)

        # Get the three blocks from upper rows
        phi = ms[..., :n, :n]                                                   # (..., n, n)
        gamma1 = ms[..., :n, n:n+m]                                             # (..., n, m)
        gamma2 = ms[..., :n, n+m:]                                              # (..., n, m)

    if is_state_preserved:
        ad = phi                                                                # (..., n, n)
        bd = gamma1 - gamma2                                                    # (..., n, m)
        b_tilded = gamma2                                                       # (..., n, m)
        cd = c                                                                  # (..., o, n)
        dd = d                                                                  # (..., o, m)
    else:
        ad = phi                                                                # (..., n, n)
        bd = gamma1 - gamma2 + phi @ gamma2                                     # (..., n, m)
        b_tilded = None
        cd = c                                                                  # (..., o, n)
        dd = d + c @ gamma2                                                     # (..., o, m)

    sysd = StateSpace(A=ad, B=bd, B_tilde=b_tilded, C=cd, D=dd)
    return sysd
