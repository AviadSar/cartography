import torch
from torch import Tensor
from torch.utils.data.sampler import Sampler

from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union

T_co = TypeVar('T_co', covariant=True)


class EvenRandomSampler(Sampler[int]):
    r"""Samples elements evenly-randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """
    data_source: Sized
    replacement: bool

    def __init__(self, data_source: Sized, dividends: int = 1, granularity: int = 1, replacement: bool = False,
                 num_samples: Optional[int] = None, generator=None) -> None:
        super().__init__(data_source)
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator
        self.dividends = dividends
        self.granularity = granularity

        if not isinstance(self.replacement, bool):
            raise TypeError("replacement should be a boolean value, but got "
                            "replacement={}".format(self.replacement))

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        num_sectors = self.dividends // self.granularity
        len_sector = n / num_sectors
        sectors = []
        for sector_idx in range(num_sectors):
            perm = torch.randperm(int((sector_idx + 1) * len_sector) - int(sector_idx * len_sector), generator=generator)
            perm += int(sector_idx * len_sector)
            sectors.append(perm.tolist())
        sampled = []
        for sector_sample_idx in range(int(len_sector // self.granularity)):
            for sector in sectors:
                sampled.extend(sector[sector_sample_idx: sector_sample_idx + self.granularity])
        for sector in sectors:
            if len(sector) > int(len_sector):
                sampled.extend(sector[-1])

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
            yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator).tolist()
        else:
            for _ in range(self.num_samples // n):
                yield from sampled
            yield from sampled[:self.num_samples % n]

    def __len__(self) -> int:
        return self.num_samples