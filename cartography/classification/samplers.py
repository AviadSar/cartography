import random
import torch
import pandas as pd
from torch import Tensor
import os
from torch.utils.data.sampler import Sampler

from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union

from cartography.selection.selection_utils import read_training_dynamics
from cartography.selection.train_dy_filtering import compute_train_dy_metrics

T_co = TypeVar('T_co', covariant=True)


class DynamicTrainingSampler(Sampler[int]):
    r"""Samples elements according to training dynamics. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """
    data_source: Sized
    replacement: bool

    def __init__(self, data_source: Sized, args=None, replacement: bool = False,
                 num_samples: Optional[int] = None, generator=None) -> None:
        super().__init__(data_source)
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator
        self.args = args

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
        args = self.args
        # def close_to_05(series):
        #     return (series.astype(float) - 0.5).abs()

        num_epochs = 0
        td_dir = os.path.join(args.td_dir, "training_dynamics")
        if os.path.exists(td_dir):
            num_epochs = len([f for f in os.listdir(td_dir) if os.path.isfile(os.path.join(td_dir, f))])

        start_dt_epoch = args.start_dt_epoch if args.start_dt_epoch is not None else 1
        if num_epochs < start_dt_epoch:
            n = len(self.data_source)
            if self.generator is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
                generator = torch.Generator()
                generator.manual_seed(seed)
            else:
                generator = self.generator
            for _ in range(self.num_samples // n):
                yield from torch.randperm(n, generator=generator).tolist()
            yield from torch.randperm(n, generator=generator).tolist()[:self.num_samples % n]
        else:
            print('TD DIR IS: {}'.format(args.td_dir), flush=True)
            training_dynamics = read_training_dynamics(args.td_dir)
            total_epochs = len(list(training_dynamics.values())[0]["logits"])
            args.burn_out = total_epochs
            args.include_ci = False
            train_dy_metrics, _ = compute_train_dy_metrics(training_dynamics, args)

            if args.metric == 'variability':
                sorted_scores = train_dy_metrics.sort_values(by=[args.metric], ascending=False)
                if train_dy_metrics[args.metric].max() == train_dy_metrics[args.metric].min():
                    train_dy_metrics['confidence'] = (train_dy_metrics['confidence'].astype(float) - 0.5).abs()
                    sorted_scores = train_dy_metrics.sort_values(by=['confidence'], ascending=True)
            elif args.metric == 'confidence':
                train_dy_metrics['confidence'] = (train_dy_metrics['confidence'].astype(float) - 0.5).abs()
                sorted_scores = train_dy_metrics.sort_values(by=[args.metric], ascending=True)
            else:
                raise ValueError('no such metric: {}'.format(args.metric))

            selected_metric = None
            if args.favored_fraction != 0:
                if args.bias is not None:
                    favored = sorted_scores.head(n=int(args.favored_fraction * len(sorted_scores)) + 1)
                    unfavored = sorted_scores.tail(n=int((1 - args.favored_fraction) * len(sorted_scores)))
                    list_favored = [favored] * args.bias
                    selected = pd.concat(list_favored + [unfavored])
                else:
                    selected = sorted_scores.head(n=int(args.favored_fraction * len(sorted_scores)) + 1)
            elif args.bias is not None:
                selected = sorted_scores
                selected_metric_min = selected[args.metric].min()
                selected_metric_max = selected[args.metric].max()
                selected_metric = ((selected[args.metric] - selected_metric_min) * (args.bias - 1) / selected_metric_max) + 1
            else:
                raise ValueError('args.favored_fraction is None, and thus no examples are selected')
            selected_guid = selected['guid'].tolist()

            for i in range(5):
                print('example data: \n', selected.iloc[i])
                print('example: \n', self.data_source[selected_guid[i]])

            if selected_metric is not None:
                for _ in range(self.num_samples):
                    yield from random.choices(selected_guid, weights=selected_metric, k=self.num_samples)
            else:
                random.shuffle(selected_guid)
                for _ in range(len(selected_guid)):
                    yield from selected_guid

    def __len__(self) -> int:
        return self.num_samples


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
                sampled.append(sector[-1])

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