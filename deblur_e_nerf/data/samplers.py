import torch


class UniformSampler(torch.utils.data.IterableDataset):
    def __init__(self, low, high, size, dtype=None, generator=None):
        super().__init__()
        self.low = low
        self.high = high
        self.size = size
        self.dtype = dtype
        self.generator = generator

    def __iter__(self):
        while True:
            yield (
                (self.high - self.low)
                * torch.rand(self.size, dtype=self.dtype,
                             generator=self.generator)
                + self.low
            )


class TriangularSampler(torch.utils.data.IterableDataset):
    """
    Implementation Reference:
        https://en.wikipedia.org/wiki/Triangular_distribution#Generating_triangular-distributed_random_variates
    """
    def __init__(self, low, high, size, mode, dtype=None, generator=None):
        super().__init__()
        for val in ( low, high, mode ):
            assert isinstance(val, (int, float))
        assert low <= mode <= high

        self.low = low
        self.high = high
        self.size = size
        self.mode = mode
        self.dtype = dtype
        self.generator = generator

        self.mode_cum_prob = (mode - low) / (high - low)
        self.k1 = (high - low) * (mode - low)
        self.k2 = (high - low) * (high - mode)

    def __iter__(self):
        while True:
            sample = torch.rand(self.size, dtype=self.dtype,
                                generator=self.generator)
            sample = torch.where(
                sample <= self.mode_cum_prob,
                self.low + torch.sqrt(sample * self.k1),
                self.high - torch.sqrt((1 - sample) * self.k2)
            )
            yield sample


class DiracDeltaSampler(torch.utils.data.IterableDataset):
    def __init__(self, center, size, dtype=None):
        super().__init__()
        self.center = center
        self.size = size
        self.dtype = dtype

    def __iter__(self):
        while True:
            size = self.size
            if isinstance(size, int):
                size = (size, )
            yield torch.full(size, self.center, dtype=self.dtype)
