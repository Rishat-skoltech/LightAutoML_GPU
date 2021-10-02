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
       `dask_cudf.Series` targets."""
    _init_checks = ()
    _data_checks = ()
    _concat_checks = ()
    _dataset_type = 'DaskCudfDataset'

    def __init__(self, data: Optional[DataFrame] = None,
                 roles: Optional[RolesDict] = None,
                 task: Optional[Task] = None,
                 index_ok: bool = False,
                  **kwargs: Series):
        """Dataset that contains `dask_cudf.core.DataFrame` and
           `dask_cudf.core.Series` target

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
                    kwargs[k] = data[f]
                    roles[f] = DropRole()
        if not index_ok:
            size = len(data.index)
            data['index'] = data.index
            mapping = dict(zip( data.index.compute().values_host,np.arange(size) ))
            data['index'] = data['index'].map(mapping).persist()
            data = data.set_index('index', drop=True, sorted=True)
            data = data.persist()
            for val in kwargs:
                col_name = kwargs[val].name
                kwargs[val] = kwargs[val].reset_index(drop=False)
                kwargs[val]['index'] = kwargs[val]['index'].map(mapping).persist()
                kwargs[val] = kwargs[val].set_index('index', drop=True, sorted=True)[col_name]

        self._initialize(task, **kwargs)
        if data is not None:
            self.set_data(data, None, roles)

    #think about doing set_index here instead of the constructor
    '''def set_data(self, data: DataFrame, features: None, roles: RolesDict):
        """Inplace set data, features, roles for empty dataset.

        Args:
            data: Table with features.
            features: `None`, just for same interface.
            roles: Dict with roles.

        """
        if isinstance(data, pd.DataFrame):
            data = cudf.DataFrame(data)
        elif isinstance(data, cudf.DataFrame):
            pass
        else:
            raise ValueError('Data type must be either pd.DataFrame or cudf.DataFrame.')
        super().set_data(data, features, roles)'''

    @staticmethod
    def _hstack(datasets: Sequence[DataFrame]) -> DataFrame:
        """Define how to concat features arrays.

        Args:
            datasets: Sequence of tables.

        Returns:
            concatenated table.

        """
        return dask_cudf.concat(datasets, axis=1).persist()

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
        self.data = self.data.persist()
        # handle dates types
        self.data = self.data.map_partitions(self._convert_datetime,
                                         date_columns, meta=self.data).persist()
        for i in date_columns:
            self.dtypes[i] = np.datetime64

    @staticmethod
    def _get_rows(data: DataFrame, k) -> FrameOrSeries:
        """Define how to get rows slice.

        Args:
            data: Table with data.
            k: Sequence of `int` indexes or `int`.

        Returns:
            Sliced rows.

        """

        return data.loc[k].persist()

    def to_cudf(self) -> CudfDataset:
        """Convert to class:`CudfDataset`.

        Returns:
            Same dataset in class:`CudfDataset` format.
        """
        data = None
        if self.data is not None:
            data = self.data.compute()
        roles = self.roles
        task = self.task

        params = dict(((x, self.__dict__[x].compute())\
                      for x in self._array_like_attrs))
        return CudfDataset(data, roles, task, **params)

    def to_numpy(self) -> 'NumpyDataset':
        """Convert to class:`NumpyDataset`.

        Returns:
            Same dataset in class:`NumpyDataset` format.

        """

        return self.to_cudf().to_numpy()

    def to_cupy(self) -> 'CupyDataset':
        """Convert dataset to cupy.

        Returns:
            Same dataset in CupyDataset format

        """

        return self.to_cudf().to_cupy()

    def to_pandas(self) -> 'PandasDataset':
        """Convert dataset to pandas.

        Returns:
            Same dataset in PandasDataset format

        """

        return self.to_cudf().to_pandas()

    def to_daskcudf(self) -> 'DaskCudfDataset':
        """Empty method to return self

        Returns:
            self
        """

        return self
        
    @staticmethod
    def _hstack(datasets: Sequence[DataFrame]) -> DataFrame:
        """Define how to concat features arrays.

        Args:
            datasets: Sequence of tables.

        Returns:
            concatenated table.

        """
        cols = []
        for i, data in enumerate(datasets):
            cols.extend(data.columns)

        #for data in datasets:
        #    print(data.compute())

        res = dask_cudf.concat(datasets, axis=1)
        mapper = dict(zip(np.arange(len(cols)), cols))
        res = res.rename(columns=mapper)
        return res
        
    @staticmethod
    def from_dataset(dataset: 'DaskCudfDataset') -> 'DaskCudfDataset':
        """Convert DaskCudfDataset to DaskCudfDatset
        (for now, later we add  to_daskcudf() to other classes
        using from_pandas and from_cudf.

        Returns:
            Converted to pandas dataset.

        """
        return dataset.to_daskcudf()

    @property
    def shape(self) -> Tuple[Optional[int], Optional[int]]:
        """Get size of 2d feature matrix.

        Returns:
            Tuple of 2 elements.

        """
        rows, cols = self.data.shape[0].compute(), len(self.features)
        return rows, cols
