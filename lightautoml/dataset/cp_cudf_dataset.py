"""Internal representation of dataset in cudf formats."""

from typing import Union, Sequence, List, Optional, TypeVar, Tuple, Any
from copy import copy
import numpy as np
import cupy as cp
import cudf

from cudf.core.dataframe import DataFrame
from cudf.core.series import Series

from cupyx.scipy import sparse

from .base import LAMLDataset, RolesDict, IntIdx, valid_array_attributes, array_attr_roles
from .roles import ColumnRole, NumericRole, DropRole
from ..tasks.base import Task
from .np_pd_dataset import NumpyDataset, PandasDataset

NpFeatures = Union[Sequence[str], str, None]
NpRoles = Union[Sequence[ColumnRole], ColumnRole, RolesDict, None]

DenseSparseArray = Union[cp.ndarray, sparse.csr_matrix]
FrameOrSeries = Union[DataFrame, Series]
Dataset = TypeVar('Dataset', bound=LAMLDataset)

class CudfDataset(PandasDataset):
    """Dataset that contains `cudf.core.dataframe.DataFrame` features and
       ` cudf.core.series.Series` targets."""
    _init_checks = ()
    _data_checks = ()
    _concat_checks = ()
    _dataset_type = 'CudfDataset'

    def __init__(self, data: Optional[DataFrame] = None,
                 roles: Optional[RolesDict] = None, task: Optional[Task] = None,
                 **kwargs: Series):
        """Create dataset from `cudf.core.dataframe.DataFrame` and
           ` cudf.core.series.Series`

        Args:
            data: Table with features.
            features: features names.
            roles: Roles specifier.
            task: Task specifier.
            **kwargs: Series, array like attrs target, group etc...

        """
        if roles is None:
            roles = {}
        # parse parameters
        # check if target, group etc .. defined in roles
        for f in roles:
            for k, r in zip(valid_array_attributes, array_attr_roles):
                if roles[f].name == r:
                    kwargs[k] = data[f].reset_index(drop=True)
                    roles[f] = DropRole()
        self._initialize(task, **kwargs)
        for k in kwargs:
            self.__dict__[k] = cudf.Series(kwargs[k])
        if data is not None:
            self.set_data(data, None, roles)

    def set_data(self, data: DataFrame, features: None, roles: RolesDict):
        """Inplace set data, features, roles for empty dataset.

        Args:
            data: Table with features.
            features: `None`, just for same interface.
            roles: Dict with roles.

        """
        super(CudfDataset.__bases__[0], self).set_data(data, features, roles)
        self._check_dtype()

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
        self.data = self.data.reset_index(drop=True)
        # do we need to reset_index ?? If yes - drop for Series attrs too
        # case to check - concat pandas dataset and from numpy to pandas dataset
        # TODO: Think about reset_index here
        # self.data.reset_index(inplace=True, drop=True)

        # handle dates types
        self.data = self._convert_datetime(self.data, date_columns)
        
        for i in date_columns:
            self.dtypes[i] = np.datetime64

    def _convert_datetime(self, data: DataFrame,
                          date_cols: List[str]) -> DataFrame:
        """Convert the listed columns of the DataFrame to DateTime type
           according to the defined roles.

        Args:
            data: Table with features.
            date_cols: Table column names that need to be converted.

        Returns:
            Data converted to datetime format from roles.

        """
        for i in date_cols:
            dt_role = self.roles[i]
            if not data.dtypes[i] is np.datetime64:
                if dt_role.unit is None:
                    data[i] = cudf.to_datetime(data[i], format=dt_role.format,
                                              origin=dt_role.origin, cache=True)
                else:
                    data[i] = cudf.to_datetime(data[i], format=dt_role.format,
                                              unit=dt_role.unit,
                                              origin=dt_role.origin, cache=True)
        return data

