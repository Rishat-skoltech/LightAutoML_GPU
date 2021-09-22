"""Numeric features transformers."""

from typing import Union

import numpy as np
import cupy as cp
import dask.array as da
import dask.dataframe as dd
import cudf

from .base import LAMLTransformer
from ..dataset.np_pd_dataset import PandasDataset, NumpyDataset
from ..dataset.cp_cudf_dataset import CudfDataset, CupyDataset
from ..dataset.daskcudf_dataset import DaskCudfDataset
from ..dataset.roles import NumericRole, CategoryRole
from .numeric import numeric_check

# type - something that can be converted to pandas dataset
CupyTransformable = Union[NumpyDataset, PandasDataset,
                          CupyDataset, CudfDataset, DaskCudfDataset]
GpuDataset = Union[CupyDataset, CudfDataset, DaskCudfDataset]


class NaNFlags_gpu(LAMLTransformer):
    """Create NaN flags."""
    _fit_checks = (numeric_check,)
    _transform_checks = ()
    _fname_prefix = 'nanflg'

    def __init__(self, nan_rate: float = .005):
        """

        Args:
            nan_rate: Nan rate cutoff.

        """
        self.nan_rate = nan_rate

    def fit_cupy(self, dataset: CupyTransformable):
        """Extract nan flags.

        Args:
            dataset: Pandas/Cudf or Numpy/Cupy dataset of categorical features.

        Returns:
            self.

        """
        #maybe we don't need this
        # set transformer names and add checks
        for check_func in self._fit_checks:
            check_func(dataset)
        # set transformer features

        # convert to accepted dtype and get attributes
        dataset = dataset.to_cupy()
        data = dataset.data
        # fit ...
        ds_nan_rate = cp.isnan(data).mean(axis=0)
        self.nan_cols = [name for (name, nan_rate) in\
                 zip(dataset.features, ds_nan_rate) if nan_rate > self.nan_rate]
        self._features = list(self.nan_cols)

        return self

    def fit_daskcudf(self, dataset: DaskCudfDataset):
        """Extract nan flags.

        Args:
            dataset: DaskCudf dataset of categorical features.

        Returns:
            self.

        """
        ds_nan_rate = dataset.data.isna().mean().compute().values
        self.nan_cols = [name for (name, nan_rate) in\
                 zip(dataset.features, ds_nan_rate) if nan_rate > self.nan_rate]
        self._features = list(self.nan_cols)

        return self

    def transform_cupy(self, dataset: CupyTransformable) -> CupyDataset:
        """Transform - extract null flags.

        Args:
            dataset: Cupy or Cudf dataset of categorical features.

        Returns:
            Cupy dataset with encoded labels.

        """
        # convert to accepted dtype and get attributes
        nans = dataset[:, self.nan_cols].data

        # transform
        new_arr = nans.isna().astype(cp.float32)
        # create resulted
        output = dataset.empty()
        output.set_data(new_arr, self.features, NumericRole(np.float32))

        return output

    def transform_daskcudf(self, dataset: DaskCudfDataset) -> DaskCudfDataset:
        """Transform - extract null flags.

        Args:
            dataset: DaskCudf dataset of categorical features.

        Returns:
            DaskCudf dataset with encoded labels.

        """
        new_data = dataset.data[self.nan_cols].isna().astype(np.float32)

        output = dataset.empty()

        output.set_data(new_data, self.features, NumericRole(np.float32))
        return output

    def fit(self, dataset: GpuDataset):
        """Extract nan flags.

        Args:
            dataset: Pandas/Cudf or Numpy/Cupy or DaskCudf dataset of categorical features.

        Returns:
            self.

        """
        assert isinstance(dataset , GpuDataset.__args__),\
               'NanFlags_gpu can do `fit` only with CupyDataset, CudfDataset, DaskCudfDataset'

        super().fit(dataset)

        if isinstance(dataset, DaskCudfDataset):
            return self.fit_daskcudf(dataset)

        else:
            return self.fit_cupy(dataset)

        return self

    def transform(self, dataset: GpuDataset) -> GpuDataset:
        """Transform - extract null flags.

        Args:
            dataset: Cudf or Cudf or DaskCudf dataset of categorical features.

        Returns:
            Respective dataset with encoded labels.

        """
        assert isinstance(dataset , GpuDataset.__args__),\
               'NanFlags_gpu can do `transform` only with CupyDataset, CudfDataset, DaskCudfDataset'

        super().transform(dataset)

        if isinstance(dataset, DaskCudfDataset):
            return self.transform_daskcudf(dataset)
        else:
            return self.transform_cupy(dataset)


class FillnaMedian_gpu(LAMLTransformer):
    """Fillna with median."""
    _fit_checks = (numeric_check,)
    _transform_checks = ()
    _fname_prefix = 'fillnamed'
    
    def fit_daskcudf(self, dataset: DaskCudfDataset):
        """Estimate medians.

        Args:
            dataset: DaskCudf dataset of categorical features.

        Returns:
            self.

        """
        data = dataset.data.astype(cp.float32).to_dask_array(lengths=True)
        self.meds = da.median(data, axis=0).compute()
        self.meds[cp.isnan(self.meds)] = 0
        self.meds = cudf.Series(self.meds, index=dataset.data.columns).astype(cp.float32)

        return self

    def fit_cupy(self, dataset: CupyTransformable):
        """Estimate medians.

        Args:
            dataset: Cudf or Cupy dataset of categorical features.

        Returns:
            self.

        """
        # set transformer features

        # convert to accepted dtype and get attributes
        dataset = dataset.to_cupy()
        data = dataset.data

        self.meds = cp.nanmedian(data, axis=0)
        self.meds[cp.isnan(self.meds)] = 0

        return self

    def transform_daskcudf(self, dataset: DaskCudfDataset) -> DaskCudfDataset:
        """Transform - fillna with medians.

        Args:
            dataset: DaskCudf dataset of categorical features.

        Returns:
            DaskCudf dataset with encoded labels.

        """
        data = dataset.data
        new_data = data.fillna(self.meds).astype(np.float32)

        output = dataset.empty()
        output.set_data(new_data, self.features, NumericRole(np.float32))
        return output

    def transform_cupy(self, dataset: CupyTransformable) -> CupyDataset:
        """Transform - fillna with medians.

        Args:
            dataset: Cudf or Cupy dataset of categorical features.

        Returns:
            Cupy dataset with encoded labels.

        """
        # convert to accepted dtype and get attributes
        data = dataset.data
        # transform
        new_data = cp.where(cp.isnan(data.values), self.meds, data.values)
        new_data = cudf.DataFrame(new_data, index=data.index, columns=self.features)
        # create resulted
        output = dataset.empty()

        #TODO
        # CHECK IF np.float32 OK OR USE cp.float32 INSTEAD
        output.set_data(new_data, self.features, NumericRole(np.float32))

        return output


    def fit(self, dataset: GpuDataset):
        """Estimate medians.

        Args:
            dataset: Cupy or Cudf or DaskCudf dataset of categorical features.

        Returns:
            self.

        """
        assert isinstance(dataset , GpuDataset.__args__),\
               'FillnaMedian_gpu can do `fit` only with CupyDataset, CudfDataset, DaskCudfDataset'

        super().fit(dataset)

        if isinstance(dataset, DaskCudfDataset):
            return self.fit_daskcudf(dataset)

        else:
            return self.fit_cupy(dataset)

        return self

    def transform(self, dataset: GpuDataset) -> GpuDataset:
        """Transform - fillna with medians.

        Args:
            dataset: Cupy or Cudf or DaskCudf dataset of categorical features.

        Returns:
            Respective dataset with encoded labels.

        """
        assert isinstance(dataset , GpuDataset.__args__),\
               'FillnaMedian_gpu can do `transform` only with CupyDataset, CudfDataset, DaskCudfDataset'

        super().transform(dataset)

        if isinstance(dataset, DaskCudfDataset):
            return self.transform_daskcudf(dataset)
        else:
            return self.transform_cupy(dataset)


class FillInf_gpu(LAMLTransformer):
    """Fill inf with nan to handle as nan value."""
    _fit_checks = (numeric_check,)
    _transform_checks = ()
    _fname_prefix = 'fillinf'

    def inf_to_nan(self, data: cudf.DataFrame) -> cudf.DataFrame:
        """Converts inf to nan on cudf.Dataframe"""

        output = cp.where(cp.isinf(data.values), cp.nan, data.values)

        #if data is single columns should be Series probably
        return cudf.DataFrame(output, columns=self.features, index=data.index)

    def transform_daskcudf(self, dataset: DaskCudfDataset) -> DaskCudfDataset:
        """Replace inf to nan.

        Args:
            dataset: DaskCudf dataset of categorical features.

        Returns:
            DaskCudf dataset with encoded labels.

        """
        new_data = dataset.data.map_partitions(self.inf_to_nan, meta=cudf.DataFrame(columns=self.features))

        output = dataset.empty()

        output.set_data(new_data, self.features, NumericRole(np.float32))
        return output

    def transform_cupy(self, dataset: CupyTransformable) -> CupyDataset:
        """Replace inf to nan.

        Args:
            dataset: Pandas/Cudf or Numpy/Cupy dataset of categorical features.

        Returns:
            Cupy dataset with encoded labels.

        """
        # convert to accepted dtype and get attributes
        #dataset = dataset.to_cupy()
        data = dataset.data
        new_data = self.inf_to_nan(data)

        # create resulted
        output = dataset.empty()

        #TODO
        # CHECK IF np.float32 OK OR USE cp.float32 INSTEAD
        output.set_data(new_data, self.features, NumericRole(np.float32))

        return output

    def transform(self, dataset: GpuDataset) -> GpuDataset:
        """Replace inf to nan.

        Args:
            dataset: Pandas/Cudf or Numpy/Cupy or DaskCudf dataset of categorical features.

        Returns:
            Respective dataset with encoded labels.

        """
        assert isinstance(dataset , GpuDataset.__args__),\
               'FillInf_gpu can do `transform` only with CupyDataset, CudfDataset, DaskCudfDataset'

        super().transform(dataset)

        if isinstance(dataset, DaskCudfDataset):
            return self.transform_daskcudf(dataset)
        else:
            return self.transform_cupy(dataset)


class LogOdds_gpu(LAMLTransformer):
    """Convert probs to logodds."""
    _fit_checks = (numeric_check,)
    _transform_checks = ()
    _fname_prefix = 'logodds'

    def num_to_logodds(self, data: cudf.DataFrame) -> cudf.DataFrame:
        """Converts num to logodds for cudf.DataFrame"""
        output = cp.clip(data.values, 1e-7, 1-1e-7)
        output = cp.log(output / (1 - output))

        return cudf.DataFrame(output, columns=self.features, index=data.index)

    def transform_cupy(self, dataset: CupyTransformable) -> CupyDataset:
        """Transform - convert num values to logodds.

        Args:
            dataset: Pandas/Cudf or Numpy/Cupy dataset of categorical features.

        Returns:
            Cupy dataset with encoded labels.

        """
        # transform
        # TODO: maybe np.exp and then cliping and logodds?
        data = self.num_to_logodds(dataset.data)

        # create resulted
        output = dataset.empty()

        # TODO: CHECK IF np.float32 OK OR USE cp.float32 INSTEAD
        output.set_data(data, self.features, NumericRole(np.float32))
        return output

    def transform_daskcudf(self, dataset: DaskCudfDataset) -> DaskCudfDataset:
        """Transform - convert num values to logodds.

        Args:
            dataset: DaskCudf dataset of categorical features.

        Returns:
            DaskCudf dataset with encoded labels.

        """
        new_data = dataset.data.map_partitions(self.num_to_logodds, meta=cudf.DataFrame(columns=self.features))

        output = dataset.empty()
        output.set_data(new_data, self.features, NumericRole(np.float32))
        return output

    def transform(self, dataset: GpuDataset) -> GpuDataset:
        """Transform - convert num values to logodds.

        Args:
            dataset: Pandas/Cudf or Numpy/Cupy or DaskCudf dataset of categorical features.

        Returns:
            Respective dataset with encoded labels.

        """
        assert isinstance(dataset , GpuDataset.__args__),\
               'LogOdds_gpu can do `transform` only with CupyDataset, CudfDataset, DaskCudfDataset'

        super().transform(dataset)

        if isinstance(dataset, DaskCudfDataset):
            return self.transform_daskcudf(dataset)
        else:
            return self.transform_cupy(dataset)


class StandardScaler_gpu(LAMLTransformer):
    """Classic StandardScaler."""

    _fit_checks = (numeric_check,)
    _transform_checks = ()
    _fname_prefix = 'scaler'

    def fit_daskcudf(self, dataset: DaskCudfDataset):
        """Estimate means and stds.

        Args:
            dataset: DaskCudf dataset of categorical features.

        Returns:
            self.

        """
        self.means = dataset.data.mean(skipna=True).compute().values
        self.stds = dataset.data.std(skipna=True).compute().values
        return self

    def fit_cupy(self, dataset: CupyTransformable):
        """Estimate means and stds.

        Args:
            dataset: Cupy or Cudf dataset of categorical features.

        Returns:
            self.

        """
        # convert to accepted dtype and get attributes
        dataset = dataset.to_cupy()
        data = dataset.data

        self.means = cp.nanmean(data, axis=0)
        self.stds = cp.nanstd(data, axis=0)
        # Fix zero stds to 1
        self.stds[(self.stds == 0) | cp.isnan(self.stds)] = 1

        return self

    def standardize(self, data):
        output = (data.values - self.means ) / self.stds
        return cudf.DataFrame(output, columns=self.features, index=data.index)

    def transform_daskcudf(self, dataset: DaskCudfDataset) -> DaskCudfDataset:
        """Scale test data.

        Args:
            dataset: DaskCudf dataset of numeric features.

        Returns:
            DaskCudf dataset with encoded labels.

        """
        new_data = dataset.data.map_partitions(self.standardize, meta=cudf.DataFrame(columns=self.features))
        output = dataset.empty()
        output.set_data(new_data, self.features, NumericRole(np.float32))
        return output

    def transform_cupy(self, dataset: CupyTransformable) -> CupyDataset:
        """Scale test data.

        Args:
            dataset: Pandas/Cudf or Numpy/Cupy dataset of numeric features.

        Returns:
            Cupy dataset with encoded labels.

        """
        # convert to accepted dtype and get attributes
        data = dataset.data

        # transform
        new_data = (data.values - self.means) / self.stds
        new_data = cudf.DataFrame(new_data, index=data.index, columns=self.features)
        # create resulted
        output = dataset.empty()

        # TODO: CHECK IF np.float32 OK OR USE cp.float32 INSTEAD
        output.set_data(data, self.features, NumericRole(np.float32))
        return output

    def fit(self, dataset: GpuDataset):
        """Estimate means and stds.

        Args:
            dataset: Cupy or Cudf or DaskCudf dataset of categorical features.

        Returns:
            self.

        """
        assert isinstance(dataset , GpuDataset.__args__),\
               'StandardScaler_gpu can do `fit` only with CupyDataset, CudfDataset, DaskCudfDataset'

        super().fit(dataset)

        if isinstance(dataset, DaskCudfDataset):
            return self.fit_daskcudf(dataset)

        else:
            return self.fit_cupy(dataset)

        return self

    def transform(self, dataset: GpuDataset) -> GpuDataset:
        """Scale test data.

        Args:
            dataset: Pandas/Cudf or Numpy/Cupy or DaskCudf dataset of numeric features.

        Returns:
            Respective dataset with encoded labels.

        """
        assert isinstance(dataset , GpuDataset.__args__),\
               'StandardScaler_gpu can do `transform` only with CupyDataset, CudfDataset, DaskCudfDataset'

        super().transform(dataset)

        if isinstance(dataset, DaskCudfDataset):
            return self.transform_daskcudf(dataset)
        else:
            return self.transform_cupy(dataset)


class QuantileBinning_gpu(LAMLTransformer):
    """Discretization of numeric features by quantiles."""
    _fit_checks = (numeric_check,)
    _transform_checks = ()
    _fname_prefix = 'qntl'

    def __init__(self, nbins: int = 10):
        """

        Args:
            nbins: maximum number of bins.

        """
        self.nbins = nbins

    def fit(self, dataset: GpuDataset):
        """Estimate bins borders.

        Args:
            dataset: Cupy or Cudf or DaskCudf dataset of numeric features.

        Returns:
            self.

        """
        assert isinstance(dataset , GpuDataset.__args__), 'QuantileBinning_gpu works only with CupyDataset, CudfDataset, DaskCudfDataset'

        super().fit(dataset)

        if isinstance(dataset, DaskCudfDataset):
            return self.fit_daskcudf(dataset)

        else:
            return self.fit_cupy(dataset)

    def fit_cupy(self, dataset: CupyTransformable):
        """Estimate bins borders.

        Args:
            dataset: Cupy or Cudf dataset of numeric features.

        Returns:
            self.

        """
        # convert to accepted dtype and get attributes
        dataset = dataset.to_cupy()
        data = dataset.data

        sl = cp.isnan(data)
        grid = cp.linspace(0, 1, self.nbins + 1)[1:-1]
        self.bins = []

        for n in range(data.shape[1]):
            q = cp.quantile(data[:, n][~sl[:, n]], q=grid)
            q = cp.unique(q)
            self.bins.append(q)
        return self

    def fit_daskcudf(self, dataset: DaskCudfDataset):
        """Estimate bins borders.

        Args:
            dataset: DaskCudf dataset of numeric features.

        Returns:
            self.

        """
        data = dataset.data
        grid = np.linspace(0, 1, self.nbins + 1)[1:-1]
        self.bins = []
        for col in data.columns:
            q = data[col].dropna().quantile(grid)
            q = q.unique().astype(cp.float32)
            self.bins.append(q.compute().values)
        return self

    def transform(self, dataset: GpuDataset) -> GpuDataset:
        """Apply bin borders.

        Args:
            dataset: Pandas/Cudf or Numpy/Cupy or DaskCudf dataset of numeric features.

        Returns:
            Respective dataset with encoded labels.

        """
        assert isinstance(dataset , GpuDataset.__args__),\
               'QuantileBinning_gpu works only with CupyDataset, CudfDataset, DaskCudfDataset'

        super().transform(dataset)

        if isinstance(dataset, DaskCudfDataset):
            return self.transform_daskcudf(dataset)
        else:
            return self.transform_cupy(dataset)

    def transform_cupy(self, dataset: CupyTransformable) -> CupyDataset:
        """Apply bin borders.

        Args:
            dataset: Pandas/Cudf or Numpy/Cupy dataset of numeric features.

        Returns:
            Cupy dataset with encoded labels.

        """
        # convert to accepted dtype and get attributes
        data = dataset.data.values

        # transform
        sl = cp.isnan(data)

        new_data = cp.zeros(data.shape, dtype=cp.int32)

        for n, b in enumerate(self.bins):
            new_data[:, n] = cp.searchsorted(b, cp.where(sl[:, n], cp.inf, data[:, n])) + 1

        new_data = cp.where(sl, 0, new_data)
        # create resulted
        output = dataset.empty()
        new_data = cudf.DataFrame(new_data, index=dataset.data.index, columns=self.features)
        output.set_data(new_data, self.features, CategoryRole(np.int32, label_encoded=True))
        return output

    def digitize(self, data):
        cp_data = data.values
        sl = cp.isnan(cp_data)
        new_data = cp.zeros(cp_data.shape, dtype=cp.int32)
        for n, b in enumerate(self.bins):
            new_data[:, n] = cp.searchsorted(b, cp.where(sl[:, n], cp.inf, cp_data[:, n])) + 1

        new_data = cp.where(sl, 0, new_data)
        output = cudf.DataFrame(new_data, columns = self.features, index=data.index)
        return output

    def transform_daskcudf(self, dataset: DaskCudfDataset) -> DaskCudfDataset:
        """Apply bin borders.

        Args:
            dataset: DaskCudf dataset of numeric features.

        Returns:
            DaskCudf dataset with encoded labels.

        """
        new_data = dataset.data.map_partitions(self.digitize,
                                              meta=cudf.DataFrame(columns=self.features)).persist()
        output = dataset.empty()
        output.set_data(new_data, self.features, CategoryRole(np.int32, label_encoded=True))
        return output

