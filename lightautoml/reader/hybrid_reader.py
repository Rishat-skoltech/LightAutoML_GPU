"""Cudf reader."""

from typing import Any, Union, Dict, Sequence, TypeVar, Optional
from copy import deepcopy

import numpy as np
from time import perf_counter

import pandas as pd

from .base import PandasToPandasReader, Reader
from .cudf_reader import CudfReader
from .daskcudf_reader import DaskCudfReader
from ..tasks import Task

from ..dataset.roles import DropRole
from ..dataset.cp_cudf_dataset import CudfDataset
from ..dataset.np_pd_dataset import PandasDataset
from ..dataset.daskcudf_dataset import DaskCudfDataset

LAMLDataset = Union[CudfDataset, PandasDataset, DaskCudfDataset]

from ..utils.logging import get_logger

from joblib import Parallel, delayed

logger = get_logger(__name__)


class HybridReader(CudfReader):
    """
    Reader to convert :class:`~cudf.core.DataFrame` to
    AutoML's :class:`~lightautoml.dataset.cp_cudf_dataset.CudfDataset`.
    Stages:

        - Drop obviously useless features.
        - Convert roles dict from user format to automl format.
        - Simple role guess for features without input role.
        - Create cv folds.
        - Create initial PandasDataset.
        - Optional: advanced guessing of role and handling types.

    """

    def __init__(self, task: Task, num_cpu_readers: int, num_gpu_readers: int, output: str, gpu_ratio: int,
                 advanced_roles: bool = True, npartitions: int = 1, *args: Any, **kwargs: Any):
        """

        Args:
            task: Task object.

        """
        super().__init__(task, *args, *kwargs)
        #self.task = task
        self.num_cpu_readers = num_cpu_readers
        self.num_gpu_readers = num_gpu_readers
        self.output = output
        self.gpu_ratio = gpu_ratio
        self.advanced_roles = advanced_roles
        self.npartitions = npartitions

        self.args = args
        self.params = kwargs

    def fit_read(self, train_data: pd.DataFrame, features_names: Any = None,
                 roles = None,
                 **kwargs: Any) -> LAMLDataset:
        """Get dataset with initial feature selection.

        Args:
            train_data: Input data.
            features_names: Ignored. Just to keep signature.
            roles: Dict of features roles in format
              ``{RoleX: ['feat0', 'feat1', ...], RoleY: 'TARGET', ....}``.
            **kwargs: Can be used for target/group/weights.

        Returns:
            Dataset with selected features.

        """
        start = perf_counter()
        parsed_roles, kwargs = self._prepare_roles_and_kwargs(roles, train_data, **kwargs)

        if self.num_gpu_readers == 0:
            assert self.num_cpu_readers != 0, 'You need at least 1 reader'
            self.gpu_ratio = 0
        elif self.num_cpu_readers == 0:
            self.gpu_ratio = 1

        train_columns = train_data.columns.difference([self.target])
        num_readers = self.num_gpu_readers + self.num_cpu_readers
        num_features = len(train_columns) - 1
        gpu_num_cols = int(num_features*self.gpu_ratio)
        cpu_num_cols = num_features - gpu_num_cols


        single_gpu_num_cols = 0
        single_cpu_num_cols = 0

        if self.num_gpu_readers != 0:
            single_gpu_num_cols = int(gpu_num_cols/self.num_gpu_readers)
        if self.num_cpu_readers != 0:
            single_cpu_num_cols = int(cpu_num_cols/self.num_cpu_readers)

        div = []
        for i in range(self.num_gpu_readers):
            div.append((i+1)*single_gpu_num_cols)
        for i in range(self.num_cpu_readers):
            div.append(gpu_num_cols + (i+1)*single_cpu_num_cols)

        div = div[:-1]
        idx = np.split(np.arange(num_features), div)
        idx = [x for x in idx if len(x) > 0]
        names = [[train_columns[x] for x in y] for y in idx]
        readers = []
        dev_num = 0

        #assert about max number of gpus here, don't forget
        for i in range(self.num_gpu_readers):
            readers.append(CudfReader(self.task, dev_num, *self.args, **self.params, advanced_roles=self.advanced_roles))
            dev_num += 1
        for i in range(self.num_cpu_readers):
            readers.append(PandasToPandasReader(self.task, *self.args, **self.params, advanced_roles=self.advanced_roles))

        for i, reader in enumerate(readers):
            names[i].append(self.target)

        def call_reader(reader, *args, **kwargs):
            reader.fit_read(*args, **kwargs)
            output_roles = reader.roles
            for feat in reader.dropped_features:
                output_roles[feat] = DropRole()
            return output_roles

        output_roles = None
        with Parallel(n_jobs=num_readers, prefer='processes', backend='loky', max_nbytes=None) as p:
        #with Parallel(n_jobs=num_readers, prefer='threads', max_nbytes=None) as p:
            output_roles = p(delayed(call_reader)(reader, train_data[name], target=train_data[self.target]) for (reader, name) in zip(readers, names))

        final_roles = {}
        final_reader = None
        for role in output_roles:
            final_roles.update(role)

        print(perf_counter() - start, "hybrid finished like")
        if self.output == 'gpu':
            final_reader = CudfReader(self.task, 0, *self.args, **self.params, advanced_roles=False)
        elif self.output == 'cpu':
            final_reader = PandasToPandasReader(self.task, *self.args, **self.params, advanced_roles=False)
        elif self.output == 'mgpu':
            final_reader = DaskCudfReader(self.task, *self.args, **self.params, advanced_roles=False, npartitions=self.npartitions)

        output = final_reader.fit_read(train_data, roles=final_roles, roles_parsed=True, target=train_data[self.target])

        return output
