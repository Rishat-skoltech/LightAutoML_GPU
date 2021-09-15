"""Internal representation of dataset in cudf formats."""

from typing import Union, Sequence, List, Optional, TypeVar, Tuple
from copy import copy
import numpy as np
import cupy as cp
import cudf
import pandas as pd

from cudf.core.dataframe import DataFrame
from cudf.core.series import Series

from cupyx.scipy import sparse

from .base import LAMLDataset, RolesDict, IntIdx, valid_array_attributes, array_attr_roles
from .roles import ColumnRole, NumericRole, DropRole
from ..tasks.base import Task
from .np_pd_dataset import NumpyDataset, PandasDataset, CSRSparseDataset

NpFeatures = Union[Sequence[str], str, None]
NpRoles = Union[Sequence[ColumnRole], ColumnRole, RolesDict, None]

DenseSparseArray = Union[cp.ndarray, sparse.csr_matrix]
FrameOrSeries = Union[DataFrame, Series]
Dataset = TypeVar('Dataset', bound=LAMLDataset)

class CupyDataset(NumpyDataset):
    """Dataset that contains `cupy.ndarray` features and
       ` cupy.ndarray` targets."""

    def _check_dtype(self):
        """Check if dtype in ``.set_data`` is ok and cast if not.

        Raises:
            AttributeError: If there is non-numeric type in dataset.

        """
        # dtypes = list(set(map(lambda x: x.dtype, self.roles.values())))
        dtypes = list(set([i.dtype for i in self.roles.values()]))
        self.dtype = cp.find_common_type(dtypes, [])

        for f in self.roles:
            self._roles[f].dtype = self.dtype

        assert cp.issubdtype(self.dtype, cp.number), \
               'Support only numeric types in Cupy dataset.'

        if self.data.dtype != self.dtype:
            self.data = self.data.astype(self.dtype)

    def __init__(self, data: Optional[DenseSparseArray],
                 features: NpFeatures = (), roles: NpRoles = None,
                 task: Optional[Task] = None, **kwargs: np.ndarray):
        """Create dataset from numpy/cupy arrays.

        Args:
            data: 2d array of features.
            features: Features names.
            roles: Roles specifier.
            task: Task specifier.
            **kwargs: Named attributes like target, group etc ..

        Note:
            For different type of parameter feature there is different behavior:

                - list, should be same len as data.shape[1]
                - None - automatic set names like feat_0, feat_1 ...
                - Prefix - automatic set names like Prefix_0, Prefix_1 ...

            For different type of parameter feature there is different behavior:

                - list, should be same len as data.shape[1].
                - None - automatic set NumericRole(np.float32).
                - ColumnRole - single role.
                - dict.

        """
        self._initialize(task, **kwargs)
        for k in kwargs:
            self.__dict__[k] = cp.asarray(kwargs[k])
        if data is not None:
            self.set_data(data, features, roles)

    def set_data(self, data: DenseSparseArray, features: NpFeatures = (),
                 roles: NpRoles = None):
        """Inplace set data, features, roles for empty dataset.

        Args:
            data: 2d np.ndarray/cp.array of features.
            features: features names.
            roles: Roles specifier.

        Note:
            For different type of parameter feature there is different behavior:

                - List, should be same len as data.shape[1]
                - None - automatic set names like feat_0, feat_1 ...
                - Prefix - automatic set names like Prefix_0, Prefix_1 ...

            For different type of parameter feature there is different behavior:

                - List, should be same len as data.shape[1].
                - None - automatic set NumericRole(cp.float32).
                - ColumnRole - single role.
                - dict.

        """
        assert data is None or type(data) is np.ndarray or type(data)\
               is cp.ndarray, 'Cupy dataset support only np.ndarray/cp.ndarray features'

        if type(data) == np.ndarray:
            data = cp.asarray(data)
        super(CupyDataset.__bases__[0], self).set_data(data, features, roles)

        if self.target is not None:
            self.target = cp.asarray(self.target)
        if self.folds is not None:
            self.folds = cp.asarray(self.folds)
        if self.weights is not None:
            self.weights = cp.asarray(self.weights)
        if self.group is not None:
            self.group = cp.asarray(self.group)

        self._check_dtype()

    @staticmethod
    def _hstack(datasets: Union[Sequence[np.ndarray],
                Sequence[cp.ndarray]]) -> cp.ndarray:
        """Concatenate function for cupy arrays.

        Args:
            datasets: Sequence of np.ndarray/cp.ndarray.

        Returns:
            Stacked features array.

        """
        return cp.hstack(datasets)

    def to_numpy(self) -> NumpyDataset:
        """Convert to numpy.

        Returns:
            Numpy dataset
        """

        assert all([self.roles[x].name == 'Numeric'\
                   for x in self.features]),\
                   'Only numeric data accepted in numpy dataset'

        data = None if self.data is None else cp.asnumpy(self.data)

        roles = self.roles
        features = self.features
        # target and etc ..
        params = dict(((x, cp.asnumpy(self.__dict__[x]))\
                       for x in self._array_like_attrs))
        task = self.task

        return NumpyDataset(data, features, roles, task, **params)

    def to_cupy(self) -> 'CupyDataset':
        """Empty method to convert to cupy.

        Returns:
            Same CupyDataset.
        """

        return self

    def to_pandas(self) -> PandasDataset:
        """Convert to PandasDataset.

        Returns:
            Same dataset in PandasDataset format.
        """

        return self.to_numpy().to_pandas()

    def to_csr(self) -> CSRSparseDataset:
        """Convert to csr.

        Returns:
            Same dataset in CSRSparseDatatset format.
        """

        return self.to_numpy().to_csr()

    def to_cudf(self) -> 'CudfDataset':
        """Convert to CudfDataset.

        Returns:
            Same dataset in CudfDataset format.
        """
        # check for empty case
        data = None if self.data is None else cudf.DataFrame()
        if data is not None:
            data_gpu = cudf.DataFrame()
            for i, col in enumerate(self.features):
                data_gpu[col] = cudf.Series(self.data[:,i], nan_as_null=False)
            data = data_gpu
        roles = self.roles
        # target and etc ..
        params = dict(((x, cudf.Series(self.__dict__[x]))\
                      for x in self._array_like_attrs))
        task = self.task

        return CudfDataset(data, roles, task, **params)

    @staticmethod
    def from_dataset(dataset: Dataset) -> 'CupyDataset':
        """Convert random dataset to cupy.

        Returns:
            Cupy dataset.

        """
        return dataset.to_cupy()

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
        if data is not None:
            self.set_data(data, None, roles)

    @property
    def roles(self) -> RolesDict:
        """Roles dict."""
        return copy(self._roles)

    @roles.setter
    def roles(self, val: NpRoles):
        """Define how to set roles.

        Args:
            val: Roles.

        Note:
            There is different behavior for different type of val parameter:

                - `List` - should be same len as ``data.shape[1]``.
                - `None` - automatic set ``NumericRole(np.float32)``.
                - ``ColumnRole`` - single role for all.
                - ``dict``.

        """
        if type(val) is dict:
            self._roles = dict(((x, val[x]) for x in self.features))
        elif type(val) is list:
            self._roles = dict(zip(self.features, val))
        else:
            role = NumericRole(np.float32) if val is None else val
            self._roles = dict(((x, role) for x in self.features))

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

    @staticmethod
    def _hstack(datasets: Sequence[DataFrame]) -> DataFrame:
        """Define how to concat features arrays.

        Args:
            datasets: Sequence of tables.

        Returns:
            concatenated table.

        """
        return cudf.concat(datasets, axis=1)

    @staticmethod
    def _get_rows(data: DataFrame, k: IntIdx) -> FrameOrSeries:
        """Define how to get rows slice.

        Args:
            data: Table with data.
            k: Sequence of `int` indexes or `int`.

        Returns:
            Sliced rows.

        """

        return data.iloc[k]

    @staticmethod
    def _get_cols(data: DataFrame, k: IntIdx) -> FrameOrSeries:
        """Define how to get cols slice.

        Args:
            data: Table with data.
            k: Sequence of `int` indexes or `int`

        Returns:
           Sliced cols.

        """
        return data.iloc[:, k]

    @classmethod
    def _get_2d(cls, data: DataFrame, k: Tuple[IntIdx, IntIdx]) -> FrameOrSeries:
        """Define 2d slice of table.

        Args:
            data: Table with data.
            k: Sequence of `int` indexes or `int`.

        Returns:
            2d sliced table.

        """
        rows, cols = k
        return data.iloc[rows, cols]

    @staticmethod
    def _set_col(data: DataFrame, k: int, val: Union[Series, np.ndarray]):
        """Inplace set column value to `cudf.DataFrame`.

        Args:
            data: Table with data.
            k: Column index.
            val: Values to set.

        """
        data.iloc[:, k] = val

    def to_numpy(self) -> NumpyDataset:
        """Convert to class:`NumpyDataset`.

        Returns:
            Same dataset in class:`NumpyDataset` format.

        """
        # check for numeric types
        dtypes = list(set([i.dtype for i in self.roles.values()]))
        self.dtype = cp.find_common_type(dtypes, [])

        for f in self.roles:
            self._roles[f].dtype = self.dtype

        assert cp.issubdtype(self.dtype, cp.number),\
               'Support only numeric types in Cupy dataset.'
        # check for empty
        data = None if self.data is None else cp.asnumpy(self.data.values)
        roles = self.roles
        features = self.features
        # target and etc ..
        params = dict(((x, cp.asnumpy(self.__dict__[x].values))\
                      for x in self._array_like_attrs))
        task = self.task
        return NumpyDataset(data, features, roles, task, **params)

    def to_cupy(self) -> CupyDataset:
        """Convert to class:`NumpyDataset`.

        Returns:
            Same dataset in class:`NumpyDataset` format.

        """
        # check for numeric types
        dtypes = list(set([i.dtype for i in self.roles.values()]))
        self.dtype = cp.find_common_type(dtypes, [])

        for f in self.roles:
            self._roles[f].dtype = self.dtype

        assert cp.issubdtype(self.dtype, cp.number),\
               'Support only numeric types in Cupy dataset.'

        # check for empty
        data = None if self.data is None else cp.asarray(self.data.values)
        roles = self.roles
        features = self.features
        # target and etc ..
        params = dict(((x, self.__dict__[x].values)\
                      for x in self._array_like_attrs))
        task = self.task

        return CupyDataset(data, features, roles, task, **params)

    def to_pandas(self) -> PandasDataset:
        """Convert dataset to pandas.

        Returns:
            Same dataset in PandasDataset format

        """
        data = self.data.to_pandas()
        roles = self.roles
        task = self.task

        params = dict(((x, pd.Series(cp.asnumpy(self.__dict__[x].values)))\
                      for x in self._array_like_attrs))

        return PandasDataset(data, roles, task, **params)

    def to_cudf(self) -> 'CudfDataset':
        """Empty method to return self

        Returns:
            self
        """

        return self

    @staticmethod
    def from_dataset(dataset: Dataset) -> 'CudfDataset':
        """Convert random dataset to cudf dataset.

        Returns:
            Converted to cudf dataset.

        """
        return dataset.to_cudf()

    def nan_rate(self) -> int:
        """Counts overall number of nans in dataset.

        Returns:
            Number of nans.

        """
        return (len(self.data) - self.data.count()).sum()
