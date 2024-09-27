"""
Adapted from `pypose.optim.GaussNewton` & `pypose.optim.LevenbergMarquardt`

Modifications:
    1. Compute jacobian using model-defined `jacobian()` or `param_jacobian()`
       function, if provided. The former does not suffer from memory spikes
       when concatenating jacobians for different parameters.
    2. Delete local variables that are no longer used to save memory

Reference:
    https://github.com/pypose/pypose/blob/main/pypose/optim/optimizer.py
"""
import torch
import pypose as pp


def has_method(obj, method_name):
    return hasattr(obj, method_name) and callable(getattr(obj, method_name))


class GaussNewton(pp.optim.GaussNewton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def step(self, input, target=None, weight=None):
        for pg in self.param_groups:
            weight = self.weight if weight is None else weight
            R = list(self.model(input, target))

            # Modification
            if has_method(self.model.model, "jacobian"):
                J = self.model.model.jacobian(input)
            else:
                if has_method(self.model.model, "param_jacobian"):
                    J = self.model.model.param_jacobian(input)
                else:
                    J = pp.optim.functional.modjac(self.model, input=(input, target), flatten=False, **self.jackwargs)
                params = dict(self.model.named_parameters())
                params_values = tuple(params.values())
                J = [self.model.flatten_row_jacobian(Jr, params_values) for Jr in J]

            for i in range(len(R)):
                R[i], J[i] = self.corrector[0](R = R[i], J = J[i]) if len(self.corrector) ==1 \
                    else self.corrector[i](R = R[i], J = J[i])
            R, weight, J = self.model.normalize_RWJ(R, weight, J)
            A, b = (J, -R) if weight is None else (weight @ J, -weight @ R)
            # Modification
            del J, R, weight
            D = self.solver(A = A, b = b.view(-1, 1))
            # Modification
            del A, b
            self.last = self.loss if hasattr(self, 'loss') \
                        else self.model.loss(input, target)
            self.update_parameter(params = pg['params'], step = D)
            # Modification
            del D
            self.loss = self.model.loss(input, target)
        return self.loss
    

class LevenbergMarquardt(pp.optim.LevenbergMarquardt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def step(self, input, target=None, weight=None):
        for pg in self.param_groups:
            weight = self.weight if weight is None else weight
            R = list(self.model(input, target))
            
            # Modification
            if has_method(self.model.model, "jacobian"):
                J = self.model.model.jacobian(input)
            else:
                if has_method(self.model.model, "param_jacobian"):
                    J = self.model.model.param_jacobian(input)
                else:
                    J = pp.optim.functional.modjac(self.model, input=(input, target), flatten=False, **self.jackwargs)
                params = dict(self.model.named_parameters())
                params_values = tuple(params.values())
                J = [self.model.flatten_row_jacobian(Jr, params_values) for Jr in J]

            for i in range(len(R)):
                R[i], J[i] = self.corrector[0](R = R[i], J = J[i]) if len(self.corrector) ==1 \
                    else self.corrector[i](R = R[i], J = J[i])
            R, weight, J = self.model.normalize_RWJ(R, weight, J)

            self.last = self.loss = self.loss if hasattr(self, 'loss') \
                                    else self.model.loss(input, target)
            J_T = J.T @ weight if weight is not None else J.T
            A, self.reject_count = J_T @ J, 0
            # Modification
            del weight
            A.diagonal().clamp_(pg['min'], pg['max'])
            while self.last <= self.loss:
                A.diagonal().add_(A.diagonal() * pg['damping'])
                try:
                    D = self.solver(A = A, b = -J_T @ R.view(-1, 1))
                except Exception as e:
                    print(e, "\nLinear solver failed. Breaking optimization step...")
                    break
                self.update_parameter(pg['params'], D)
                self.loss = self.model.loss(input, target)
                self.strategy.update(pg, last=self.last, loss=self.loss, J=J, D=D, R=R.view(-1, 1))
                if self.last < self.loss and self.reject_count < self.reject: # reject step
                    self.update_parameter(params = pg['params'], step = -D)
                    self.loss, self.reject_count = self.last, self.reject_count + 1
                else:
                    break
        return self.loss
