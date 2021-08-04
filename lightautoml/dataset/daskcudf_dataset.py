"""Internal representation of dataset in dask_cudf format."""

from typing import Union, Sequence, Optional, TypeVar, Tuple

import numpy as np
import cupy as cp
import dask_cudf

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

class DaskCudfDataset(CudfDataset):
    """Dataset that contains `dask_cudf.core.DataFrame` features and
       `dask_cudf..Series` targets."""
    _init_checks = ()
    _data_checks = ()
    _concat_checks = ()
    _dataset_type = 'DaskCudfDataset'

    def __init__(self, data: Optional[DataFrame] = None,
                 roles: Optional[RolesDict] = None, task: Optional[Task] = None,
                 **kwargs: Series):
        """Create dataset from `dask_cudf.core.DataFrame` and `dask_cudf.core.Series`

        Args:
            data: Table with features.
            features: features names.
            roles: Roles specifier.
            task: Task specifier.
            npartitions: Number of partitions of the table.
            **kwargs: Series, array like attrs target, group etc...

        """
        if roles is None:
            roles = {}
        # parse parameters
        # check if target, group etc .. defined in roles
        for f in roles:
            for k, r in zip(valid_array_attributes, array_attr_roles):
                if roles[f].name == r:
                    kwargs[k] = data[f].reset_index(drop=True).persist()
                    roles[f] = DropRole()
        self._initialize(task, **kwargs)
        #for k in kwargs:
        #    self.__dict__[k] = dask_cudf.Series(kwargs[k])
        
        size = len(data.index)
        data['index'] = data.index
        mapping = dict(zip( data.index.compute().values_host,np.arange(size) ))
        data['index'] = data['index'].map(mapping).persist()  
        data = data.set_index('index', drop=True, sorted=True)
        
        if data is not None:
            self.set_data(data, None, roles)


    @staticmethod
    def _get_rows(data: DataFrame, k) -> FrameOrSeries:
        """Define how to get rows slice.

        Args:
            data: Table with data.
            k: Sequence of `int` indexes or `int`.

        Returns:
            Sliced rows.

        """

        return data.loc[k]#.compute()

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

        # handle dates types
        self.data = self.data.map_partitions(self._convert_datetime,
                                             date_columns, meta=self.data).persist()
        for i in date_columns:
            self.dtypes[i] = np.datetime64
            
    @property
    def shape(self) -> Tuple[Optional[int], Optional[int]]:
        """Get size of 2d feature matrix.

        Returns:
            Tuple of 2 elements.

        """
        rows, cols = self.data.shape[0].compute(), len(self.features)
        return rows, cols
            
            
