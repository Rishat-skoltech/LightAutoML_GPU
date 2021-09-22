"""Datetime features transformers."""

from collections import OrderedDict
from typing import Union, Sequence, List, Optional

import holidays
import numpy as np
import cupy as cp

import cudf
import pandas as pd


from .base import LAMLTransformer
from ..dataset.base import LAMLDataset
from ..dataset.cp_cudf_dataset import CudfDataset, CupyDataset
from ..dataset.daskcudf_dataset import DaskCudfDataset
from ..dataset.roles import NumericRole, CategoryRole, ColumnRole

from .datetime import date_attrs, datetime_check

DatetimeCompatible = Union[CudfDataset]
GpuDataset = Union[CupyDataset, CudfDataset, DaskCudfDataset]

class TimeToNum_gpu(LAMLTransformer):
    """
    Basic conversion strategy, used in selection one-to-one transformers.
    Datetime converted to difference
    with basic_date (``basic_date == '2020-01-01'``).
    """
    def __init__(self):

        self.basic_time = '2020-01-01'
        self.basic_interval = 'D'

        self._fname_prefix = 'dtdiff'
        self._fit_checks = (datetime_check,)
        self._transform_checks = ()

    def transform(self, dataset: GpuDataset) -> GpuDataset:
        """Transform dates to numeric differences with base date.

        Args:
            dataset:  Cupy or Cudf or DaskCudf dataset with datetime columns.

        Returns:
            Respective dataset of numeric features.

        """
        assert isinstance(dataset , GpuDataset.__args__),\
               'TimeToNum_gpu works only with CupyDataset, CudfDataset, DaskCudfDataset'

        super().transform(dataset)

        if isinstance(dataset, DaskCudfDataset):
            return self.transform_daskcudf(dataset)
        else:
            return self.transform_cupy(dataset)

        return output

    def standardize_date(self, data: cudf.DataFrame,
                    mean: np.datetime64, std: np.timedelta64) -> cudf.DataFrame:
        """Perform starndartizatino of the date.

        Args:
            data: DataFrame with the dates.
            mean: Data to center to.
            std: Time delta to normalize to.

        Returns:
            DataFrame with starndardized dates.

        """
        output = (data.astype(int) - mean) / std

        return output

    def transform_daskcudf(self, dataset: DaskCudfDataset) -> DaskCudfDataset:
        """Transform dates to numeric differences with base date.

        Args:
            dataset:  DaskCudf dataset with datetime columns.

        Returns:
            DaskCudf dataset of numeric features.

        """
        data = dataset.data

        time_diff = cudf.DatetimeIndex(pd.date_range(self.basic_time,
                                       periods=1, freq='d')).astype(int)[0]

        #shouldn't hardcode this,
        #should take units from dataset.roles(but its unit is none currently)
        timedelta = np.timedelta64(1, self.basic_interval)/np.timedelta64(1, 'ns')

        new_data = data.map_partitions(self.standardize_date, time_diff, timedelta, meta=cudf.DataFrame(columns=data.columns))

        output = dataset.empty()
        output.set_data(new_data, self.features, NumericRole(cp.float32))
        return output

    def transform_cupy(self, dataset: DatetimeCompatible) -> CupyDataset:
        """Transform dates to numeric differences with base date.

        Args:
            dataset:  Cudf dataset with datetime columns.

        Returns:
            Cupy dataset of numeric features.

        """
        data = dataset.data

        time_diff = cudf.DatetimeIndex(pd.date_range(self.basic_time,
                                       periods=1, freq='d')).astype(int)[0]

        #shouldn't hardcode this,
        #should take units from dataset.roles(but its unit is none currently)
        timedelta = np.timedelta64(1, self.basic_interval)/np.timedelta64(1, 'ns')

        new_data = self.standardize_date(data, time_diff, timedelta)

        output = dataset.empty()
        output.set_data(new_data, self.features, NumericRole(cp.float32))

        return output


class BaseDiff_gpu(LAMLTransformer):
    """
    Basic conversion strategy, used in selection one-to-one transformers.
    Datetime converted to difference with basic_date.

    """
    basic_interval = 'D'

    _fname_prefix = 'basediff'
    _fit_checks = (datetime_check,)
    _transform_checks = ()

    @property
    def features(self) -> List[str]:
        """List of features."""
        return self._features

    def __init__(self, base_names: Sequence[str], diff_names: Sequence[str],
                 basic_interval: Optional[str] = 'D'):
        """

        Args:
            base_names: Base date names.
            diff_names: Difference date names.
            basic_interval: Time unit.

        """
        self.base_names = base_names
        self.diff_names = diff_names
        self.basic_interval = basic_interval

    def fit(self, dataset: LAMLDataset) -> 'LAMLTransformerGPU':
        """Fit transformer and return it's instance.

        Args:
            dataset: Dataset to fit on.

        Returns:
            self.

        """
        assert isinstance(dataset , GpuDataset.__args__),\
               'BaseDiff_gpu works only with CupyDataset, CudfDataset, DaskCudfDataset'

        self._features = []
        for col in self.base_names:
            self._features.extend(['basediff_{0}__{1}'.format(col, x) for x in self.diff_names])

        for check_func in self._fit_checks:
            check_func(dataset)
        return self

    def transform_cupy(self, dataset: DatetimeCompatible) -> CupyDataset:
        """Transform dates to numeric differences with base date.

        Args:
            dataset: Cupy or Cudf dataset with datetime columns.

        Returns:
            CupyDataset of numeric features.

        """
        # convert to accepted format and get attributes
        dataset = dataset.to_cudf()
        data = dataset.data[self.diff_names]
        base_cols = dataset.data[self.base_names]
        feats_block = []

        time_delta = cudf.Series(np.timedelta64(1, self.basic_interval)).repeat(len(data)).reset_index()[0]

        for col in base_cols.columns:
            new_arr = cp.empty(data.shape)
            for n, i in enumerate(data.columns):
                new_arr[:,n] = ((data[i] - base_cols[col])/time_delta).values.astype(cp.float32)
            feats_block.append(new_arr)
        feats_block = cp.concatenate(feats_block, axis=1)

        # create resulted
        output = dataset.empty().to_cupy()
        output.set_data(feats_block, self.features, NumericRole(dtype=cp.float32))

        return output

    def standardize_date_concat(self, data, std):
        feats_block = []
        for col in self.base_names:

            output = (data[self.diff_names].astype(int).values.T - data[col].astype(int).values) / std
            feats_block.append(output.T)

        return cudf.DataFrame(cp.concatenate(feats_block, axis=1), columns=self.features)

    def transform_daskcudf(self, dataset: DaskCudfDataset) -> DaskCudfDataset:
        """Transform dates to numeric differences with base date.

        Args:
            dataset: DaskCudf dataset with datetime columns.

        Returns:
            DaskDataset of numeric features.

        """
        data = dataset.data

        #shouldn't hardcode this,
        #should take units from dataset.roles
        #(but its unit is none currently)
        timedelta = np.timedelta64(1, self.basic_interval)/np.timedelta64(1, 'ns')

        new_data = data.map_partitions(self.standardize_date_concat, timedelta, meta=cudf.DataFrame(columns=self.features))

        output = dataset.empty()
        output.set_data(new_data, self.features, NumericRole(cp.float32))
        return output

    def transform(self, dataset: GpuDataset) -> GpuDataset:
        """Transform dates to numeric differences with base date.

        Args:
            dataset: Cupy or Cudf or DaskCudf dataset with datetime columns.

        Returns:
            Respective dataset numeric features.

        """
        assert isinstance(dataset , GpuDataset.__args__),\
               'BaseDiff_gpu works only with CupyDataset, CudfDataset, DaskCudfDataset'

        super().transform(dataset)

        if isinstance(dataset, DaskCudfDataset):
            return self.transform_daskcudf(dataset)
        else:
            return self.transform_cupy(dataset)


class DateSeasons_gpu(LAMLTransformer):
    """
    Basic conversion strategy, used in selection one-to-one transformers.
    Datetime converted to difference with basic_date.
    """

    _fname_prefix = 'season'
    _fit_checks = (datetime_check,)
    _transform_checks = ()

    @property
    def features(self) -> List[str]:
        """List of features names."""
        return self._features

    def __init__(self, output_role: Optional[ColumnRole] = None):
        """

        Args:
            output_role: Which role to assign for input features.

        """
        self.output_role = output_role
        if output_role is None:
            self.output_role = CategoryRole(cp.int32)

    def fit(self, dataset: LAMLDataset) -> 'LAMLTransformerGPU':
        """Fit transformer and return it's instance.

        Args:
            dataset: LAMLDataset to fit on.

        Returns:
            self.

        """
        assert isinstance(dataset , GpuDataset.__args__),\
               'DateSeasons_gpu works only with CupyDataset, CudfDataset, DaskCudfDataset'

        for check_func in self._fit_checks:
            check_func(dataset)

        feats = dataset.features
        roles = dataset.roles
        self._features = []
        self.transformations = OrderedDict()

        for col in feats:
            seas = roles[col].seasonality
            self.transformations[col] = seas
            for s in seas:
                self._features.append('season_{0}__{1}'.format(s, col))
            if roles[col].country is not None:
                self._features.append('season_hol__{0}'.format(col))

        return self

    def transform(self, dataset: GpuDataset) -> GpuDataset:
        """Transform dates to categories - seasons and holiday flag.

        Args:
            dataset: Cupy or Cudf or DaskCudf dataset with datetime columns.

        Returns:
            Respective dataset of numeric features.

        """
        assert isinstance(dataset , GpuDataset.__args__),\
               'DateSeasons_gpu works only with CupyDataset, CudfDataset, DaskCudfDataset'

        super().transform(dataset)

        if isinstance(dataset, DaskCudfDataset):
            return self.transform_daskcudf(dataset)
        else:
            return self.transform_cupy(dataset)

    def transform_cupy(self, dataset: DatetimeCompatible) -> CupyDataset:
        """Transform dates to categories - seasons and holiday flag.

        Args:
            dataset: Cupy or Cudf dataset with datetime columns.

        Returns:
            Cupy dataset of numeric features.

        """
        # convert to accepted format and get attributes
        dataset = dataset.to_cudf()
        df = dataset.data
        roles = dataset.roles

        new_arr = self.datetime_to_seasons(df, roles, date_attrs)

        output = dataset.empty()
        output.set_data(new_arr, self.features, self.output_role)

        return output

    def datetime_to_seasons(self, data: cudf.DataFrame, roles, _date_attrs) -> cudf.DataFrame:
        new_arr = cp.empty((data.shape[0], len(self._features)), cp.int32)
        n = 0
        for col in data.columns:
            for seas in self.transformations[col]:
                vals = getattr(data[col].dt, _date_attrs[seas]).values.astype(cp.int32)
                new_arr[:, n] = vals
                n += 1

            if roles[col].country is not None:
                # get years
                years = cp.unique(data[col].dt.year)
                hol = holidays.CountryHoliday(roles[col].country, years=years,
                                              prov=roles[col].prov, state=roles[col].state)
                new_arr[:, n] = data[col].isin(cudf.Series(pd.Series(hol)))
                n += 1
        return cudf.DataFrame(new_arr, index=data.index, columns=self.features)

    def transform_daskcudf(self, dataset: DaskCudfDataset) -> DaskCudfDataset:
        """Transform dates to categories - seasons and holiday flag.

        Args:
            dataset: DaskCudf dataset with datetime columns.

        Returns:
            Dask dataset of numeric features.

        """
        new_arr = dataset.data.map_partitions(self.datetime_to_seasons,
                                              dataset.roles, date_attrs,
                                              meta=cudf.DataFrame(columns = self.features))
        output = dataset.empty()
        output.set_data(new_arr, self.features, self.output_role)
        return output


