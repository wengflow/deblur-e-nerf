import torch


class OffsetGammaCorrection(torch.nn.Module):
    def __init__(
        self,
        const_scale=1.0,
        init_scale=1.0,
        init_gamma=1.0,
        init_offset=0.0
    ):
        super().__init__()
        if torch.is_tensor(const_scale):
            const_scale = const_scale.detach().clone()
        else:
            const_scale = torch.tensor(const_scale)
        
        if torch.is_tensor(init_scale):
            init_scale = init_scale.detach().clone()
        else:
            init_scale = torch.tensor(init_scale)

        if torch.is_tensor(init_gamma):
            init_gamma = init_gamma.detach().clone()
        else:
            init_gamma = torch.tensor(init_gamma)

        if torch.is_tensor(init_offset):
            init_offset = init_offset.detach().clone()
        else:
            init_offset = torch.tensor(init_offset)
        
        self.register_buffer("const_scale", const_scale, persistent=False)
        self.scale = torch.nn.parameter.Parameter(init_scale)        
        self.gamma = torch.nn.parameter.Parameter(init_gamma)
        self.offset = torch.nn.parameter.Parameter(init_offset)

    def forward(self, input):
        return self.const_scale * (
            self.scale * input.pow(self.gamma) - self.offset
        )
    
    def dense_jacobian(self, input):
        """
        Computes the partial derivative of each output element wrt. the
        specific scale, gamma & offset (scalar) parameters involved in its
        computation (i.e. dense jacobian).
        """
        dense_scale_jac = self.const_scale * input.pow(self.gamma)              # (B, C, H, W, R)
        dense_gamma_jac = self.scale * input.log() * dense_scale_jac            # (B, C, H, W, R)
        dense_offset_jac = (-self.const_scale).expand(input.shape)              # (B, C, H, W, R)

        return (dense_scale_jac, dense_gamma_jac, dense_offset_jac)
    
    def param_jacobian(self, input):
        """
        NOTE:
            Currently only supports constant scale, scale, gamma & offset
            buffer/parameters with shapes (B, 1, 1, 1, 1), (1/C, 1, 1, 1),
            (1/C, 1, 1, 1) & (1/C, 1, 1, 1) and input with shape
            (B, C, H, W, R).
        """
        assert len(self.const_scale) == self.const_scale.numel()
        assert len(self.const_scale) == len(input)

        C = input.shape[1]
        for param_name in ( "scale", "gamma", "offset" ):
            param = getattr(self, param_name)
            assert len(param) == param.numel()
            assert len(param) == 1 or len(param) == C

        # derive the (true, sparse) jacobian from the dense jacobian
        dense_scale_jac, dense_gamma_jac, dense_offset_jac = \
            self.dense_jacobian(input)                                          # (B, C, H, W, R), (B, C, H, W, R), (B, C, H, W, R)

        if len(self.scale) == 1:
            scale_jac = dense_scale_jac.view(*input.shape, *self.scale.shape)   # (B, C, H, W, R, 1, 1, 1, 1)
        else:   # elif len(self.scale) == C:
            dense_scale_jac = dense_scale_jac.transpose(0, 1)                   # (C, B, H, W, R)
            scale_jac = torch.zeros(*input.shape, *self.scale.shape,            # (B, C, H, W, R, C, 1, 1, 1)
                                    dtype=input.dtype, device=input.device)
            scale_jac[:, range(C), :, :, :, range(C), 0, 0, 0] = (              # (C, B, H, W, R)
                dense_scale_jac
            )
            del dense_scale_jac

        if len(self.gamma) == 1:
            gamma_jac = dense_gamma_jac.view(*input.shape, *self.gamma.shape)   # (B, C, H, W, R, 1, 1, 1, 1)
        else:   # elif len(self.gamma) == C:
            dense_gamma_jac = dense_gamma_jac.transpose(0, 1)                   # (C, B, H, W, R)
            gamma_jac = torch.zeros(*input.shape, *self.gamma.shape,            # (B, C, H, W, R, C, 1, 1, 1)
                                    dtype=input.dtype, device=input.device)
            gamma_jac[:, range(C), :, :, :, range(C), 0, 0, 0] = (              # (C, B, H, W, R)
                dense_gamma_jac
            )
            del dense_gamma_jac

        if len(self.offset) == 1:
            offset_jac = dense_offset_jac.view(*input.shape,                    # (B, C, H, W, R, 1, 1, 1, 1)
                                               *self.offset.shape)
        else:   # elif len(self.offset) == C:
            dense_offset_jac = dense_offset_jac.transpose(0, 1)                 # (C, B, H, W, R)
            offset_jac = torch.zeros(*input.shape, *self.offset.shape,          # (B, C, H, W, R, C, 1, 1, 1)
                                     dtype=input.dtype, device=input.device)
            offset_jac[:, range(C), :, :, :, range(C), 0, 0, 0] = (             # (C, B, H, W, R)
                dense_offset_jac
            )
            del dense_offset_jac

        return [[ scale_jac, gamma_jac, offset_jac ]]
    
    def jacobian(self, input):
        """
        NOTE:
            Currently only supports constant scale, scale, gamma & offset
            buffer/parameters with shapes (B, 1, 1, 1, 1), (1/C, 1, 1, 1),
            (1/C, 1, 1, 1) & (1/C, 1, 1, 1) and input with shape
            (B, C, H, W, R).
        """
        assert len(self.const_scale) == self.const_scale.numel()
        assert len(self.const_scale) == len(input)

        C = input.shape[1]
        for param_name in ( "scale", "gamma", "offset" ):
            param = getattr(self, param_name)
            assert len(param) == param.numel()
            assert len(param) == 1 or len(param) == C

        # derive the (true, sparse) jacobian from the dense jacobian
        dense_scale_jac, dense_gamma_jac, dense_offset_jac = \
            self.dense_jacobian(input)                                          # (B, C, H, W, R), (B, C, H, W, R), (B, C, H, W, R)

        N = input.numel()                                                       # i.e. B * C * H * W * R
        S = len(self.scale)                                                     # i.e. 1/C
        G = len(self.gamma)                                                     # i.e. 1/C
        O = len(self.offset)                                                    # i.e. 1/C
        jacobian = torch.zeros(N, S + G + O,                                    # (N, S + G + O)
                               dtype=input.dtype, device=input.device)
        
        scale_jac = jacobian[:, :S]                                             # (N, S)
        scale_jac = scale_jac.view(*input.shape, S)                             # (B, C, H, W, R, S)
        if S == 1:
            scale_jac[..., 0] = dense_scale_jac                                 # (B, C, H, W, R)
        else:   # elif S == C:
            dense_scale_jac = dense_scale_jac.transpose(0, 1)                   # (C, B, H, W, R)
            scale_jac[:, range(C), :, :, :, range(C)] = dense_scale_jac         # (C, B, H, W, R)
        del dense_scale_jac, scale_jac

        gamma_jac = jacobian[:, S:S+G]                                          # (N, G)
        gamma_jac = gamma_jac.view(*input.shape, G)                             # (B, C, H, W, R, G)
        if G == 1:
            gamma_jac[..., 0] = dense_gamma_jac                                 # (B, C, H, W, R)
        else:   # elif G == C:
            dense_gamma_jac = dense_gamma_jac.transpose(0, 1)                   # (C, B, H, W, R)
            gamma_jac[:, range(C), :, :, :, range(C)] = dense_gamma_jac         # (C, B, H, W, R)
        del dense_gamma_jac, gamma_jac

        offset_jac = jacobian[:, S+G:]                                          # (N, O)
        offset_jac = offset_jac.view(*input.shape, O)                           # (B, C, H, W, R, O)
        if O == 1:
            offset_jac[..., 0] = dense_offset_jac                               # (B, C, H, W, R)
        else:   # elif O == C:
            dense_offset_jac = dense_offset_jac.transpose(0, 1)                 # (C, B, H, W, R)
            offset_jac[:, range(C), :, :, :, range(C)] = dense_offset_jac       # (C, B, H, W, R)
        del dense_offset_jac, offset_jac

        return [ jacobian ]
