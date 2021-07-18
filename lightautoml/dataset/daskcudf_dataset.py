"""Internal representation of dataset in dask_cudf format."""

from typing import Union, Sequence, Optional, TypeVar

import numpy as np
import cupy as cp

from log_calls import record_history

from dask_cudf.core import DataFrame, Series

from cupyx.scipy import sparse

from .base import LAMLDataset, RolesDict, valid_array_attributes, array_attr_roles
from .roles import ColumnRole, DropRole
from ..tasks.base import Task
from .cp_cudf_dataset import CudfDataset

NpFeatures = Union[Sequence[str], str, None]
NpRoles = Union[Sequence[ColumnRole], ColumnRole, RolesDict, None]

DenseSparseArray = Union[cp.ndarray, sparse.csr_matrix]
FrameOrSeries = Union[DataFrame, Series]
Dataset = TypeVar('Dataset', bound=LAMLDataset)

# possible checks list
# valid shapes
# target var is ok for task
# pandas - roles for all columns are defined
# numpy - roles and features are ok each other
# numpy - roles and features are ok for data
# features names does not contain __ - it's used to split processing names

# sparse - do not replace init and set data, but move type assert in checks?

#this decorator works bad with distributed client
#@record_history(enabled=False)
class DaskCudfDataset(CudfDataset):
    """Dataset that contains `dask_cudf.core.DataFrame` features and
       `dask_cudf..Series` targets."""
    _dataset_type = 'DaskCudfDataset'

    def __init__(self, data: Optional[DataFrame] = None,
                 roles: Optional[RolesDict] = None, task: Optional[Task] = None,
                 npartitions: int = None, **kwargs: Series):
        """Create dataset from `dask_cudf.core.DataFrame` and `dask_cudf.core.Series`

        Args:
            data: Table with features.
            features: features names.
            roles: Roles specifier.
            task: Task specifier.
            npartitions: Number of partitions of the table.
            **kwargs: Series, array like attrs target, group etc...

        """
        self.npartitions = npartitions
        if roles is None:
            roles = {}
        # parse parameters
        # check if target, group etc .. defined in roles
        for f in roles:
            for k, r in zip(valid_array_attributes, array_attr_roles):
                if roles[f].name == r:
                    kwargs[k] = data[f].map_partitions(self._reset_index).persist()
                    roles[f] = DropRole()
        self._initialize(task, **kwargs)
        if data is not None:
            self.set_data(data, None, roles)


    def _check_dtype(self):
        """Check if dtype in .set_data is ok and cast if not."""
        date_columns = []

        self.dtypes = {}
        for f in self.roles:
            if self.roles[f].name == 'Datetime':
                date_columns.append(f)
            else:
                self.dtypes[f] = self.roles[f].dtype

        #UNCOMMENT THIS AFTER 21.08 RELEASE
        #self.data = self.data.astype(self.dtypes)
        # do we need to reset_index ?? If yes - drop for Series attrs too
        # case to check - concat pandas dataset and from numpy to pandas dataset
        # TODO: Think about reset_index here

        #self.data = self.data.reset_index(drop=True).persist()
        self.data = self.data.map_partitions(self._reset_index,
                                             meta=self.data).persist()

        # handle dates types
        self.data = self.data.map_partitions(self._convert_datetime,
                                             date_columns, meta=self.data).persist()
        for i in date_columns:
            self.dtypes[i] = np.datetime64
