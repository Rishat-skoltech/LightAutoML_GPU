"""Categorical features transformers."""

from itertools import combinations
from typing import Optional, Union, List, Sequence, cast, Tuple

from time import perf_counter

import numpy as np
import cupy as cp
import pandas as pd
from copy import copy
from cupyx import scatter_add
import cudf
import dask_cudf


from cuml.preprocessing import OneHotEncoder as one_hot

from .base import LAMLTransformer
from ..dataset.cp_cudf_dataset import CudfDataset, CupyDataset
from ..dataset.daskcudf_dataset import DaskCudfDataset
from ..dataset.roles import CategoryRole, NumericRole
from .categorical import categorical_check, oof_task_check,\
                     multiclass_task_check, encoding_check

GpuNumericalDataset = Union[CupyDataset, CudfDataset, DaskCudfDataset]


class LabelEncoder_gpu(LAMLTransformer):
    """Simple LabelEncoder in order of frequency.

    Labels are integers from 1 to n. Unknown category encoded as 0.
    NaN is handled as a category value.

    """
    _fit_checks = (categorical_check,)
    _transform_checks = ()
    _fname_prefix = 'le'
    _fillna_val = 0

    def __init__(self, subs: Optional[int] = None, random_state: int = 42):
        """

        Args:
            subs: Subsample to calculate freqs. If None - full data.
            random_state: Random state to take subsample.

        """
        self.subs = subs
        self.random_state = random_state
        self._output_role = CategoryRole(cp.int32, label_encoded=True)

    def _get_df(self, dataset: GpuNumericalDataset) -> cudf.DataFrame:
        """Get df and sample.

        Args:
            dataset: Input dataset.

        Returns:
            Subsample.

        """
        #dataset = dataset.to_cudf()
        df = dataset.data

        if self.subs is not None and df.shape[0] >= self.subs:
            subs = df.sample(n=self.subs, random_state=self.random_state)
        else:
            subs = df

        return subs

    def fit_daskcudf(self, dataset: GpuNumericalDataset):
        """Estimate label frequencies and create encoding dicts.

        Args:
            dataset: DaskCudf dataset of categorical features.

        Returns:
            self.

        """
        # convert to accepted dtype and get attributes
        roles = dataset.roles
        self.dicts = {}

        daskcudf_data = dataset.data
        
        for i in daskcudf_data.columns:
            role = roles[i]
            co = role.unknown
            cnts = daskcudf_data[i].value_counts(dropna=False).compute().reset_index() \
                .sort_values([i, 'index'], ascending=[False, True]).set_index('index')
            ids = (cnts > co)[cnts.columns[0]]
            vals = cnts[ids].index
            self.dicts[i] = cudf.Series(cp.arange(vals.shape[0], dtype=cp.int32) + 1, index=vals)
        return self

    def fit_cupy(self, dataset: GpuNumericalDataset):
        """Estimate label frequencies and create encoding dicts.

        Args:
            dataset: Cudf or Cupy dataset of categorical features.

        Returns:
            self.

        """
        # set transformer features

        # convert to accepted dtype and get attributes
        roles = dataset.roles
        subs = self._get_df(dataset)
        self.dicts = {}
        map_streams = []
        for i in range(8):
            map_streams.append(cp.cuda.stream.Stream())

        for num,i in enumerate(subs.columns):
            with map_streams[num%8]:
                role = roles[i]
                co = role.unknown
                cnts = subs[i].value_counts(dropna=False).reset_index() \
                    .sort_values([i, 'index'], ascending=[False, True]).set_index('index')
                ids = (cnts > co)[cnts.columns[0]]
                vals = cnts[ids].index
                self.dicts[i] = cudf.Series(cp.arange(vals.shape[0], dtype=cp.int32) + 1, index=vals)
        cp.cuda.Device().synchronize()
        return self

    def fit(self, dataset: GpuNumericalDataset):
        """Estimate label frequencies and create encoding dicts.

        Args:
            dataset: Cudf or Cupy or DaskCudf dataset of categorical features.

        Returns:
            self.

        """
        assert isinstance(dataset , GpuNumericalDataset.__args__),\
               'LabelEncoder_gpu can do `fit` only with CupyDataset, CudfDataset, DaskCudfDataset'

        super().fit(dataset)

        if isinstance(dataset, DaskCudfDataset):
            return self.fit_daskcudf(dataset)
        else:
            return self.fit_cupy(dataset)

    def transform(self, dataset: GpuNumericalDataset) -> GpuNumericalDataset:
        """Transform categorical dataset to int labels.

        Args:
            dataset: Cudf or Cupy or DaskCudf dataset of categorical features.

        Returns:
            Respective dataset with encoded labels.

        """
        assert isinstance(dataset , GpuNumericalDataset.__args__),\
        'LabelEncoder_gpu can do `transform` only with CupyDataset, CudfDataset, DaskCudfDataset'
        super().transform(dataset)

        if isinstance(dataset, DaskCudfDataset):
            return self.transform_daskcudf(dataset)
        else:
            return self.transform_cupy(dataset)

    def transform_daskcudf(self, dataset: DaskCudfDataset) -> DaskCudfDataset:
        """Transform categorical dataset to int labels.

        Args:
            dataset: DaskCudf dataset of categorical features.

        Returns:
            DaskCudf dataset with encoded labels.

        """
        daskcudf_data = dataset.data
        new_arr = daskcudf_data.map_partitions(self.encode_labels,
                               meta=cudf.DataFrame(columns=self.features).astype(self._output_role.dtype) ).persist()
                               
        output = dataset.empty()
        output.set_data(new_arr, self.features, self._output_role)
        return output

    def encode_labels(self, df):
        new_arr = cudf.DataFrame(index=df.index, columns=self.features)

        for n,i in enumerate(df.columns):
            out_col = new_arr.columns[n]
            # to be compatible with OrdinalEncoder
            if i in self.dicts:
                if not self.dicts[i].index.is_unique:
                    sl = df[i].isna()
                    cur_dict = self.dicts[i][self.dicts[i].index.dropna()]
                    if len(cur_dict) > 0:
                        new_arr[out_col] = df[i].map(cur_dict).fillna(self._fillna_val)
                    else:
                        if not sl.all():
                            new_arr[out_col] = cudf.Series(cp.ones(len(df[i]))*cp.nan, index=df[i].index, nan_as_null=False)
                        else:
                            new_arr[out_col][~sl] = cp.nan

                    nan_val = self.dicts[i].iloc[-1]
                    new_arr[out_col][sl] = nan_val
                else:
                    new_arr[out_col] = df[i].map(self.dicts[i]).fillna(self._fillna_val)
            else:
                new_arr[out_col] = df[i].astype(self._output_role.dtype)
        return new_arr

    def transform_cupy(self, dataset: GpuNumericalDataset) -> CupyDataset:
        """Transform categorical dataset to int labels.

        Args:
            dataset: Cudf or Cupy dataset of categorical features.

        Returns:
            Cupy dataset with encoded labels.

        """
        # convert to accepted dtype and get attributes
        #dataset = dataset.to_cudf()
        df = dataset.data

        # transform
        new_arr = self.encode_labels(df)

        # create resulted
        output = dataset.empty()
        output.set_data(new_arr, self.features, self._output_role)
        return output


class OHEEncoder_gpu(LAMLTransformer):
    """
    Simple OneHotEncoder over label encoded categories.
    """
    _fit_checks = (categorical_check, encoding_check)
    _transform_checks = ()
    _fname_prefix = 'ohe'

    @property
    def features(self) -> List[str]:
        """Features list."""
        return self._features

    def __init__(self, make_sparse: Optional[bool] = False, total_feats_cnt: Optional[int] = None,
                 dtype: type = cp.float32):
        """

        Args:
            make_sparse: Create sparse matrix.
            total_feats_cnt: Initial features number.
            dtype: Dtype of new features.

        """
        self.make_sparse = make_sparse
        self.total_feats_cnt = total_feats_cnt
        self.dtype = dtype

        if self.make_sparse is None:
            assert self.total_feats_cnt is not None,\
                   'Param total_feats_cnt should be defined if make_sparse is None'

    def fit_cupy(self, dataset: GpuNumericalDataset):
        """Calc output shapes.

        Args:
            dataset: Cudf or Cupy dataset of categorical features.

        Returns:
            self.

        """
        # set transformer names and add checks
        for check_func in self._fit_checks:
            check_func(dataset)
        # set transformer features

        # convert to accepted dtype and get attributes
        dataset = dataset.to_cupy()
        data = dataset.data
        max_idx = cp.asnumpy(data.max(axis=0))

        # infer make sparse
        if self.make_sparse is None:
            fill_rate = self.total_feats_cnt / (self.total_feats_cnt - max_idx.shape[0] + max_idx.sum())
            self.make_sparse = fill_rate < 0.2

        # create ohe

        self.ohe = one_hot(categories='auto',
                           dtype=self.dtype, sparse=self.make_sparse,
                           handle_unknown='ignore')
        self.ohe.fit(data)

        features = []
        for cats, name in zip(self.ohe.categories_, dataset.features):
            pd_cats = cats.to_pandas()
            features.extend(['ohe_{0}__{1}'.format(x, name) for x in pd_cats])
        self._features = features

        return self

    def fit_daskcudf(self, dataset: DaskCudfDataset):
        """Calc output shapes.

        Args:
            dataset: DaskCudf dataset of categorical features.

        Returns:
            self.

        """
        # set transformer names and add checks
        for check_func in self._fit_checks:
            check_func(dataset)
        # set transformer features

        # convert to accepted dtype and get attributes
        data = dataset.data

        max_idx = data.max(axis=0).compute()

        # infer make sparse
        if self.make_sparse is None:
            fill_rate = self.total_feats_cnt / (self.total_feats_cnt - max_idx.shape[0] + max_idx.sum())
            self.make_sparse = fill_rate < 0.2

        # create ohe

        self.ohe = one_hot(categories='auto',
                           dtype=self.dtype, sparse=self.make_sparse,
                           handle_unknown='ignore')

        max_val = int(max_idx.max())
        temp_data = cp.ones((len(data.columns), max_val+1)) *(max_val+1)
        for i,col in enumerate(data.columns):
            uniques = data[col].unique().compute().values
            temp_data[i][:uniques.shape[0]] = uniques

        #or Series if only one col
        temp_data = cudf.DataFrame(temp_data.T, columns=data.columns)
        self.ohe.fit(temp_data)

        features = []
        for cats, name in zip(self.ohe.categories_, dataset.features):
            pd_cats = cats.to_pandas()
            features.extend(['ohe_{0}__{1}'.format(x, name) for x in pd_cats])
        self._features = features
        #or Series if only one col

        return self

    def transform_cupy(self, dataset: GpuNumericalDataset) -> CupyDataset:
        """Transform categorical dataset to ohe.

        Args:
            dataset: Cudf or Cupy dataset of categorical features.

        Returns:
            Cupy dataset with encoded labels.

        """
        # convert to accepted dtype and get attributes
        dataset = dataset.to_cupy()
        data = dataset.data

        # transform
        data = self.ohe.transform(data)

        # create resulted
        output = dataset.empty()
        if self.make_sparse:
            output = output.to_csr()
        output.set_data(data, self.features, NumericRole(self.dtype))

        return output

    def call_ohe(self, data: cudf.DataFrame) -> cudf.DataFrame:
        """Call the fitted OHE encoder from cuml

        Args:
            data: cudf.DataFrame with data
        Returns:
            cudf.DataFrame with encoded labels
        """
        output = self.ohe.transform(data)
        return cudf.DataFrame(output, index=data.index, columns=self.features)

    def transform_daskcudf(self, dataset: DaskCudfDataset) -> DaskCudfDataset:
        """Transform categorical dataset to ohe.

        Args:
            dataset: DaskCudf dataset of categorical features.

        Returns:
            DaskCupy dataset with encoded labels.

        """
        new_data = dataset.data.map_partitions(self.call_ohe, meta=cudf.DataFrame(columns=self.features, index=dataset.data.index) )

        output = dataset.empty()

        output.set_data(new_data, self.features, NumericRole(self.dtype))
        print("############THIS IS OHE START############")
        print(new_data.compute())
        print("#############THIS IS OHE END#############")
        return output

    def fit(self, dataset: GpuNumericalDataset):
        """Calc output shapes.

        Args:
            dataset: Cudf or Cupy or DaskCudf dataset of categorical features.

        Returns:
            self.

        """
        assert isinstance(dataset , GpuNumericalDataset.__args__),\
              'OHEEncoder_gpu can do `fit` only with CupyDataset, CudfDataset, DaskCudfDataset'

        if isinstance(dataset, DaskCudfDataset):
            return self.fit_daskcudf(dataset)
        else:
            return self.fit_cupy(dataset)

    def transform(self, dataset: GpuNumericalDataset) -> GpuNumericalDataset:
        """Transform categorical dataset to ohe.

        Args:
            dataset: Cupy or Cudf or DaskCudf dataset of categorical features.

        Returns:
            Respective dataset with encoded labels.

        """
        assert isinstance(dataset , GpuNumericalDataset.__args__),\
        'OHEEncoder_gpu can do `transform` only with CupyDataset, CudfDataset, DaskCudfDataset'

        if isinstance(dataset, DaskCudfDataset):
            return self.transform_daskcudf(dataset)
        else:
            return self.transform_cupy(dataset)


class FreqEncoder_gpu(LabelEncoder_gpu):
    """
    Labels are encoded with frequency in train data.

    Labels are integers from 1 to n. Unknown category encoded as 1.
    """
    _fit_checks = (categorical_check,)
    _transform_checks = ()
    _fname_prefix = 'freq'

    _fillna_val = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: check if it is necessary to change dtype to cp.float32
        self._output_role = NumericRole(cp.float32)

    def fit_daskcudf(self, dataset: GpuNumericalDataset):
        """Estimate label frequencies and create encoding dicts.

        Args:
            dataset: DaskCudf dataset of categorical features

        Returns:
            self.

        """
        # set transformer names and add checks
        LAMLTransformer.fit(self, dataset)
        # set transformer features

        # convert to accepted dtype and get attributes
        self.dicts = {}

        daskcudf_data = dataset.data

        for i in daskcudf_data.columns:
            cnts = daskcudf_data[i].value_counts(dropna=False)
            self.dicts[i] = cnts[cnts > 1].compute()

        return self

    def fit_cupy(self, dataset: GpuNumericalDataset):
        """Estimate label frequencies and create encoding dicts.

        Args:
            dataset: Cudf or Cupy dataset of categorical features

        Returns:
            self.

        """
        # set transformer names and add checks
        LAMLTransformer.fit(self, dataset)
        # set transformer features

        # convert to accepted dtype and get attributes
        #dataset = dataset.to_cudf()
        df = dataset.data

        self.dicts = {}
        for i in df.columns:
            cnts = df[i].value_counts(dropna=False)
            self.dicts[i] = cnts[cnts > 1]
        return self


class TargetEncoder_gpu(LAMLTransformer):
    """
    Out-of-fold target encoding.

    Limitation:

        - Required .folds attribute in dataset - array of int from 0 to n_folds-1.
        - Working only after label encoding.

    """
    _fit_checks = (categorical_check, oof_task_check, encoding_check)
    _transform_checks = ()
    _fname_prefix = 'oof'

    def __init__(self, alphas: Sequence[float] = (.5, 1., 2., 5., 10., 50., 250., 1000.)):
        """

        Args:
            alphas: Smooth coefficients.

        """
        self.alphas = alphas

    @staticmethod
    def dask_add_at_1d(data: cudf.DataFrame, col: List[str], val: Union[int, str], size: int) -> cudf.DataFrame:
        """Wrapper around scatter_add for dask_cudf.DataFrame

        Args:
            data: cudf.DataFrame data
            col: str columns to work on
            val: int or str - value or column name to to the addition

        Return:
            cudf.DataFrame with results of computations
        """
        output = cp.zeros(int(size), dtype=cp.float32)
        if isinstance(val, int):
            scatter_add(output, data[col].values, val)
        else:
            scatter_add(output, data[col].values, data[val].values)
        return cudf.DataFrame([output])#, columns=np.arange(size))

    @staticmethod
    def dask_add_at_2d(data: cudf.DataFrame, cols: List[str], val: Union[int, str], shape: Tuple[int, int]) -> cudf.DataFrame:
        output = cp.zeros(shape, dtype=cp.float32)
        if isinstance(val, int):
            scatter_add(output, (data[cols[0]].values, data[cols[-1]].values), val)
        else:
            scatter_add(output, (data[cols[0]].values, data[cols[-1]].values), data[val].values)
        return cudf.DataFrame(output)#, columns=np.arange(n_folds))

    @staticmethod
    def find_candidates(data, vec_col, fold_col, oof_sum, oof_count, alphas, folds_prior):
        """Find oof candidates for metric scroing.

        Args:
            data:
            vec_col:
            fold_col:
            oof_sum:
            oof_count:
            alphas:
            folds_prior:

        Returns:
            cudf.DataFrame with output

        """
        vec = data[vec_col].values
        folds = data[fold_col].values
        candidates = ((oof_sum[vec, folds, cp.newaxis] + alphas * folds_prior[folds, cp.newaxis])
                      /(oof_count[vec, folds, cp.newaxis] + alphas)).astype(cp.float32)
        return cudf.DataFrame(candidates, index=data.index)

    @staticmethod
    def dask_binary_score_func(data: cudf.DataFrame, target_col: str) -> cudf.DataFrame:
        """Score candidates alpha with logloss metric.

        Args:
            data: Candidate oof encoders.
            target_col: column name with target.

        Returns:
            cudf.DataFrame with scores

        """
        target = data[target_col].values[:, cp.newaxis]
        #assuming that target_col is the last
        candidates = data[data.columns[:-1]].values
        scores = - (target * cp.log(candidates) + (1 - target) * cp.log(1 - candidates)).mean(axis=0)
        return cudf.DataFrame([scores])

    @staticmethod
    def dask_reg_score_func(data: cudf.DataFrame, target_col: str) -> cudf.DataFrame:
        """Score candidates alpha with mse metric.

        Args:
            data: Candidate oof encoders.
            target_col: column name with target.

        Returns:
            cudf.DataFrame with scores

        """
        target = data[target_col].values[:, cp.newaxis]
        #assuming that target_col is the last
        candidates = data[data.columns[:-1]].values
        scores = ((target - candidates) ** 2).mean(axis=0)
        return cudf.DataFrame([scores])

    @staticmethod
    def binary_score_func(candidates: cp.ndarray, target: cp.ndarray) -> int:
        """Score candidates alpha with logloss metric.

        Args:
            candidates: Candidate oof encoders.
            target: Target array.

        Returns:
            Index of best encoder.

        """
        target = target[:, cp.newaxis]
        scores = - (target * cp.log(candidates) + (1 - target) * cp.log(1 - candidates)).mean(axis=0)
        idx = scores.argmin()

        return idx

    @staticmethod
    def reg_score_func(candidates: cp.ndarray, target: cp.ndarray) -> int:
        """Score candidates alpha with mse metric.

        Args:
            candidates: Candidate oof encoders.
            target: Target array.

        Returns:
            Index of best encoder.

        """
        target = target[:, cp.newaxis]
        scores = ((target - candidates) ** 2).mean(axis=0)
        idx = scores.argmin()

        return idx

    def fit(self, dataset: GpuNumericalDataset):

        assert isinstance(dataset , GpuNumericalDataset.__args__),\
        'TargetEncoder_gpu can do `fit` only with CupyDataset, CudfDataset, DaskCudfDataset'

        super().fit_transform(dataset)

    def fit_transform(self, dataset: GpuNumericalDataset) ->GpuNumericalDataset:
        """Calc oof encoding and save encoding stats for new data.

        Args:
            dataset: Cudf or Cupy or DaskCudf dataset of
              categorical label encoded features.

        Returns:
            Respective dataset  - target encoded features.

        """
        assert isinstance(dataset , GpuNumericalDataset.__args__),\
        'TargetEncoder_gpu can do `fit_transform` only with CupyDataset, CudfDataset, DaskCudfDataset'

        if isinstance(dataset, DaskCudfDataset):
            return self.fit_transform_daskcudf(dataset)
        else:
            return self.fit_transform_cupy(dataset)

    def fit_transform_daskcudf(self, dataset: DaskCudfDataset) -> DaskCudfDataset:
        """Calc oof encoding and save encoding stats for new data.

        Args:
            dataset: DaskCudf dataset of
              categorical label encoded features.

        Returns:
            DaskCudf - target encoded features.

        """
        st = perf_counter()
        super().fit(dataset)
        print(perf_counter() - st, " superfit")
        st = perf_counter()
        score_func = self.dask_binary_score_func if dataset.task.name == 'binary'\
                                               else self.dask_reg_score_func

        alphas = cp.array(self.alphas)[cp.newaxis, :]
        self.encodings = []
        print(perf_counter() - st, "preps")
        st = perf_counter()

        prior = dataset.target.mean().compute()
        print(perf_counter() - st, "prior")
        st = perf_counter()

        target_name = dataset.target.name
        folds_name = dataset.folds.name

        daskcudf_data = dataset.data.persist()#copy(dataset.data)
        daskcudf_data[folds_name] = dataset.folds
        daskcudf_data[target_name] = dataset.target
        print(perf_counter() - st, "persisting")
        st = perf_counter()

        n_folds = int(daskcudf_data[folds_name].max().compute() + 1)

        f_sum = daskcudf_data.map_partitions(self.dask_add_at_1d, folds_name, target_name, n_folds,
                              meta=cudf.DataFrame(columns=np.arange(n_folds), dtype='f8') ).sum().compute().values
        f_count = daskcudf_data.map_partitions(self.dask_add_at_1d, folds_name, 1, n_folds,
                              meta=cudf.DataFrame(columns=np.arange(n_folds), dtype='i8')).sum().compute().values

        print(perf_counter() - st, "two map partitions with dask add")
        st = perf_counter()
        folds_prior = (f_sum.sum() - f_sum) / (f_count.sum() - f_count)
        print(perf_counter() - st, "folds_prior")
        st = perf_counter()

        oof_feats = daskcudf_data[dataset.features]#.persist()
        print(perf_counter() - st, "before cycling")
        st = perf_counter()

        for n in range(oof_feats.shape[1]):
            vec_col = daskcudf_data.columns[n]

            enc_dim = int(daskcudf_data[vec_col].max().compute() + 1)

            f_sum = daskcudf_data.map_partitions(self.dask_add_at_2d, [vec_col, folds_name],
                                                target_name, (enc_dim, n_folds),
                                                meta=cudf.DataFrame(columns=np.arange(n_folds), dtype='f8') ).compute().values
            f_count = daskcudf_data.map_partitions(self.dask_add_at_2d, [vec_col, folds_name],
                                                1, (enc_dim, n_folds),
                                                meta=cudf.DataFrame(columns=np.arange(n_folds), dtype='i8') ).compute().values

            '''f_sum_final = cp.zeros((enc_dim, n_folds))
            f_count_final = cp.zeros((enc_dim, n_folds))

            for i in range(enc_dim):
                f_sum_final[i] = f_sum[f_sum.columns][f_sum.index.values==i].sum().values
            for i in range(enc_dim):
                f_count_final[i] = f_count[f_count.columns][f_count.index.values==i].sum().values'''
            f_sum_final = f_sum.reshape((-1, enc_dim, n_folds)).sum(axis=0)
            f_count_final = f_count.reshape((-1, enc_dim, n_folds)).sum(axis=0)

            # calc total stats
            t_sum = f_sum_final.sum(axis=1, keepdims=True)
            t_count = f_count_final.sum(axis=1, keepdims=True)

            # calc oof stats
            oof_sum = t_sum - f_sum_final
            oof_count = t_count - f_count_final
            # calc candidates alpha
            candidates = daskcudf_data.map_partitions(self.find_candidates, vec_col,
                                             folds_name, oof_sum, oof_count, alphas, folds_prior).persist()

            candidates[target_name] = daskcudf_data[target_name]

            scores = candidates.map_partitions(score_func, target_name).compute().mean(axis=0).values
            idx = scores.argmin().get()

            oof_feats[vec_col] = candidates[candidates.columns[idx]]#.persist()
            # calc best encoding
            enc = ((t_sum[:, 0] + alphas[0, idx] * prior) / (t_count[:, 0] + alphas[0, idx])).astype(cp.float32)

            self.encodings.append(enc)

        print(perf_counter() - st, "after cycling")
        st = perf_counter()
        output = dataset.empty()
        self.output_role = NumericRole(cp.float32, prob=output.task.name == 'binary')
        print(perf_counter() - st, "output")
        st = perf_counter()
        output.set_data(oof_feats.persist(), self.features, self.output_role)
        print(perf_counter() - st, "setting data")
        st = perf_counter()
        return output

    def fit_transform_cupy(self, dataset: GpuNumericalDataset) -> CupyDataset:
        """Calc oof encoding and save encoding stats for new data.

        Args:
            dataset: Cudf or Cupy dataset of
              categorical label encoded features.

        Returns:
            Cupy - target encoded features.

        """
        # set transformer names and add checks
        super().fit(dataset)
        # set transformer features

        # convert to accepted dtype and get attributes
        #dataset = dataset.to_cupy()
        data = dataset.data.values

        # TODO: check if it is necessary to change dtype to cp.int32
        target = cp.asarray(dataset.target).astype(cp.int32)
        score_func = self.binary_score_func if dataset.task.name == 'binary' else self.reg_score_func

        folds = cp.asarray(dataset.folds)
        n_folds = int(folds.max() + 1)
        alphas = cp.array(self.alphas)[cp.newaxis, :]

        self.encodings = []

        prior = target.mean()

        # folds priors
        f_sum = cp.zeros(n_folds, dtype=cp.float32)
        f_count = cp.zeros(n_folds, dtype=cp.float32)

        scatter_add(f_sum, folds, target)
        scatter_add(f_count, folds, 1)

        folds_prior = (f_sum.sum() - f_sum) / (f_count.sum() - f_count)
        oof_feats = cp.zeros(data.shape, dtype=cp.float32)

        map_streams = []
        for i in range(8):
            map_streams.append(cp.cuda.stream.Stream())
        for n in range(data.shape[1]):
            with map_streams[n%8]:
                vec = data[:, n]

                # calc folds stats
                enc_dim = int(vec.max() + 1)

                f_sum = cp.zeros((enc_dim, n_folds), dtype=cp.float32)
                f_count = cp.zeros((enc_dim, n_folds), dtype=cp.float32)

                scatter_add(f_sum, (vec, folds), target)
                scatter_add(f_count, (vec, folds), 1)

                # calc total stats
                t_sum = f_sum.sum(axis=1, keepdims=True)
                t_count = f_count.sum(axis=1, keepdims=True)

                # calc oof stats
                oof_sum = t_sum - f_sum
                oof_count = t_count - f_count
                # calc candidates alpha
                candidates = ((oof_sum[vec, folds, cp.newaxis] + alphas * folds_prior[folds, cp.newaxis])
                              / (oof_count[vec, folds, cp.newaxis] + alphas)).astype(cp.float32)
                idx = score_func(candidates, target)

                # write best alpha
                oof_feats[:, n] = candidates[:, idx]
                # calc best encoding
                enc = ((t_sum[:, 0] + alphas[0, idx] * prior) / (t_count[:, 0] + alphas[0, idx])).astype(cp.float32)

                self.encodings.append(enc)
        cp.cuda.Device().synchronize()
        output = dataset.empty()
        self.output_role = NumericRole(cp.float32, prob=output.task.name == 'binary')
        output.set_data(cudf.DataFrame(oof_feats, columns = self.features, index = dataset.data.index), self.features, self.output_role)
        return output

    def transform(self, dataset: GpuNumericalDataset) -> GpuNumericalDataset:
        """Transform categorical dataset to target encoding.

        Args:
            dataset: Cudf or Cupy or DaskCudf dataset of categorical features.

        Returns:
            Respective dataset with encoded labels.

        """
        assert isinstance(dataset , GpuNumericalDataset.__args__),\
               'TargetEncoder_gpu works only with CupyDataset, CudfDataset, DaskCudfDataset'

        if isinstance(dataset, DaskCudfDataset):
            #TODO: transform for daskcudf
            raise NotImplementedError
        else:
            return self.transform_cupy(dataset)

    def transform_cupy(self, dataset: GpuNumericalDataset) -> CupyDataset:
        """Transform categorical dataset to target encoding.

        Args:
            dataset: Cudf or Cupy dataset of categorical features.

        Returns:
            Cudf dataset with encoded labels.

        """
        # checks here
        super().transform(dataset)
        # convert to accepted dtype and get attributes
        dataset = dataset.to_cupy()
        data = dataset.data

        # transform
        out = cp.zeros(data.shape, dtype=cp.float32)
        for n, enc in enumerate(self.encodings):
            out[:, n] = enc[data[:, n]]

        # create resulted
        output = dataset.empty()
        output.set_data(out, self.features, self.output_role)
        return output


class MultiClassTargetEncoder_gpu(LAMLTransformer):
    """
    Out-of-fold target encoding for multiclass task.

    Limitation:

        - Required .folds attribute in dataset - array of int from 0 to n_folds-1.
        - Working only after label encoding

    """
    _fit_checks = (categorical_check, multiclass_task_check, encoding_check)
    _transform_checks = ()
    _fname_prefix = 'multioof'

    @property
    def features(self) -> List[str]:
        return self._features

    def __init__(self, alphas: Sequence[float] = (.5, 1., 2., 5., 10., 50., 250., 1000.)):
        self.alphas = alphas

    @staticmethod
    def score_func(candidates: cp.ndarray, target: cp.ndarray) -> int:
        """


        Args:
            candidates: cp.ndarray.
            target: cp.ndarray.

        Returns:
            index of best encoder.

        """
        target = target[:, cp.newaxis, cp.newaxis]
        scores = -cp.log(cp.take_along_axis(candidates, target, axis=1)).mean(axis=0)[0]
        idx = scores.argmin()

        return idx

    @staticmethod
    def dask_add_at_2d(data, cols, val, shape):
        output = cp.zeros(shape, dtype=cp.float32)
        if isinstance(cols[0], int):
            scatter_add(output, (cols[0], data[cols[-1]].values), val)
        else:
            scatter_add(output, (data[cols[0]].values, data[cols[-1]].values), val)
        return cudf.DataFrame(output)#, columns=np.arange(n_folds))

    @staticmethod
    def dask_add_at_3d(data, cols, val, shape):
        output = cp.zeros(shape, dtype=cp.float32)
        if isinstance(cols[1], int):
            scatter_add(output, (data[cols[0]].values, cols[1], data[cols[-1]].values), val)
        else:
            scatter_add(output, (data[cols[0]].values, data[cols[1]].values, data[cols[-1]].values), val)

        output = output.reshape((shape[0], shape[1]*shape[2]))
        return cudf.DataFrame(output)#, columns=np.arange(n_folds))

    @staticmethod
    def dask_score_func(data, target_col, shape):
        target = data[target_col].values
        target = target[:, cp.newaxis, cp.newaxis]
        #the fact that target_col is the last is hardcoded here
        candidates = data[data.columns[:-1]].values
        #reshape it here
        candidates = candidates.reshape(data.shape[0], shape[0], shape[1])

        scores = -cp.log(cp.take_along_axis(candidates, target, axis=1)).mean(axis=0)[0]
        return cudf.DataFrame([scores])

    @staticmethod
    def find_candidates(data, vec_col, fold_col, oof_sum, oof_count, alphas, folds_prior):
        vec = data[vec_col].values
        folds = data[fold_col].values
        candidates = ((oof_sum[vec, :, folds, cp.newaxis] + alphas * folds_prior[folds, :, cp.newaxis])
                          / (oof_count[vec, :, folds, cp.newaxis] + alphas)).astype(cp.float32)

        candidates /= candidates.sum(axis=1, keepdims=True)
        candidates = candidates.reshape(data.shape[0], -1)
        return cudf.DataFrame(candidates, index = data.index)

    def find_prior(self, target, n_classes):
        prior = cast(cp.ndarray, cp.arange(n_classes)[:, cp.newaxis] == target).mean(axis=1)
        return cudf.DataFrame([prior])

    def fit_transform(self, dataset: GpuNumericalDataset) -> GpuNumericalDataset:
        """Estimate label frequencies and create encoding dicts.

        Args:
            dataset: Cupy or Cudf or DaskCudf dataset of categorical label encoded features.

        Returns:
            Respective dataset - target encoded features.

        """
        assert isinstance(dataset, GpuNumericalDataset.__args__),\
               'TargetEncoder_gpu works only with CupyDataset, CudfDataset, DaskCudfDataset'

        if isinstance(dataset, DaskCudfDataset):
            return self.fit_transform_daskcudf(dataset)
        else:
            return self.fit_transform_cupy(dataset)

    def fit_transform_daskcudf(self, dataset: DaskCudfDataset) -> DaskCudfDataset:
        """Estimate label frequencies and create encoding dicts.

        Args:
            dataset: DaskCudf dataset of categorical label encoded features.

        Returns:
            DaskCudfDataset - target encoded features.

        """
        # set transformer names and add checks
        for check_func in self._fit_checks:
            check_func(dataset)
        # set transformer features

        alphas = cp.array(self.alphas)[cp.newaxis, cp.newaxis, :]
        self.encodings = []

        target_name = dataset.target.name
        folds_name = dataset.folds.name

        daskcudf_data = copy(dataset.data)
        daskcudf_data[folds_name] = dataset.folds#.astype('i8')
        daskcudf_data[target_name] = dataset.target#.astype('i8')

        n_folds = int(daskcudf_data[folds_name].max().compute() + 1)
        n_classes = int(dataset.target.max().compute() + 1)
        self.n_classes = n_classes

        prior = dataset.target.map_partitions(self.find_prior, n_classes).mean(axis=0).compute().values

        f_sum = cp.zeros((n_classes, n_folds), dtype=cp.float32)
        f_count = cp.zeros((1, n_folds), dtype=cp.float32)

        f_sum = daskcudf_data.map_partitions(self.dask_add_at_2d, [target_name, folds_name],
                                      1, (n_classes, n_folds),
                                      meta=cudf.DataFrame(columns=np.arange(n_folds), dtype='f8') )
        f_count = daskcudf_data.map_partitions(self.dask_add_at_2d, [0, folds_name], 1, (1, n_folds),
                                              meta=cudf.DataFrame(columns=np.arange(n_folds), dtype='i8'))

        f_sum_final = cp.zeros((n_classes, n_folds))
        f_count_final = cp.zeros((1, n_folds))

        for i in range(n_classes):
            f_sum_final[i] = f_sum.compute()[f_sum.columns][f_sum.index.compute().values==i].sum().values

        f_count_final[0] = f_count.sum(axis=0).compute().values

        folds_prior = ((f_sum_final.sum(axis=1, keepdims=True) - f_sum_final)\
                      / (f_count_final.sum(axis=1, keepdims=True) - f_count_final)).T

        oof_feats = []

        self._features = []
        for i in dataset.features:
            for j in range(n_classes):
                self._features.append('{0}_{1}__{2}'.format('multioof', j, i))

        for n,col in enumerate(dataset.features):
            vec_col = col#daskcudf_data.columns[n]

            enc_dim = int(daskcudf_data[vec_col].max().compute() + 1)

            f_sum = daskcudf_data.map_partitions(self.dask_add_at_3d, [vec_col, target_name, folds_name],
                                    1, (enc_dim, n_classes, n_folds),
                                    meta=cudf.DataFrame(np.empty((enc_dim,n_classes*n_folds)), dtype='f8')).compute().values
            f_count = daskcudf_data.map_partitions(self.dask_add_at_3d, [vec_col, 0, folds_name],
                                      1, (enc_dim, 1, n_folds),
                                      meta=cudf.DataFrame(np.empty((enc_dim,n_folds)), dtype='i8')).compute().values

            '''f_sum_final = cp.zeros((enc_dim, n_classes*n_folds))
            f_count_final = cp.zeros((enc_dim, 1*n_folds))
            for i in range(enc_dim):
                f_sum_final[i] = f_sum.compute()[f_sum.columns][f_sum.index.compute().values==i].sum().values
            for i in range(enc_dim):
                f_count_final[i] = f_count.compute()[f_count.columns][f_count.index.compute().values==i].sum().values'''
            f_sum_final = f_sum.reshape((-1, enc_dim, n_folds*n_classes)).sum(axis=0)
            f_count_final = f_count.reshape((-1, enc_dim, n_folds)).sum(axis=0)

            f_sum_final = f_sum_final.reshape((enc_dim, n_classes, n_folds))
            f_count_final = f_count_final.reshape((enc_dim, 1, n_folds))
            t_sum = f_sum_final.sum(axis=2, keepdims=True)
            t_count = f_count_final.sum(axis=2, keepdims=True)

            oof_sum = t_sum - f_sum_final
            oof_count = t_count - f_count_final
            candidates = daskcudf_data.map_partitions(self.find_candidates, vec_col,
                                folds_name, oof_sum, oof_count, alphas, folds_prior,
                                meta=cudf.DataFrame(columns = np.arange(len(self.alphas)*n_classes), dtype='f8') )

            candidates[target_name] = daskcudf_data[target_name]

            scores = candidates.map_partitions(self.dask_score_func, target_name,
                                   (n_classes, len(self.alphas)),
                                   meta=cudf.DataFrame(columns=np.arange(len(self.alphas)), dtype='f8')).compute().mean(axis=0).values
            idx = scores.argmin().get()
            orig_cols = np.arange(idx*n_classes, (idx+1)*n_classes)
            new_cols = self.features[n*n_classes: (n+1)*n_classes]
            col_map = dict(zip(orig_cols, new_cols))
            oof_feats.append(candidates[candidates.columns[idx*n_classes:(idx+1)*n_classes]].rename( columns = col_map ) )
            enc = ((t_sum[..., 0] + alphas[0, 0, idx] * prior) / (t_count[..., 0] + alphas[0, 0, idx])).astype(cp.float32)
            enc /= enc.sum(axis=1, keepdims=True)

            self.encodings.append(enc)
        orig_cols = np.arange(n_classes*len(dataset.features))
        col_map = dict(zip(orig_cols, self.features))
        oof_feats = dask_cudf.concat(oof_feats, axis=1).rename(columns = col_map)
        output = dataset.empty()
        output.set_data(oof_feats, self.features, NumericRole(cp.float32, prob=True))
        return output

    def fit_transform_cupy(self, dataset: GpuNumericalDataset) -> CudfDataset:
        """Estimate label frequencies and create encoding dicts.

        Args:
            dataset: Cudf or Cupy dataset of categorical label encoded features.

        Returns:
            CudfDataset - target encoded features.

        """
        # set transformer names and add checks
        for check_func in self._fit_checks:
            check_func(dataset)
        # set transformer features

        # convert to accepted dtype and get attributes
        data = dataset.data.values
        target = cp.asarray(dataset.target).astype(cp.int32)
        n_classes = int(target.max() + 1)
        self.n_classes = n_classes

        folds = cp.asarray(dataset.folds)
        n_folds = int(folds.max() + 1)
        alphas = cp.array(self.alphas)[cp.newaxis, cp.newaxis, :]

        self.encodings = []
        # prior
        prior = cast(cp.ndarray, cp.arange(n_classes)[:, cp.newaxis] == target).mean(axis=1)
        # folds prior

        f_sum = cp.zeros((n_classes, n_folds), dtype=cp.float64)
        f_count = cp.zeros((1, n_folds), dtype=cp.float64)
        scatter_add(f_sum, (target, folds), 1)
        scatter_add(f_count, (0, folds), 1)

        # N_classes x N_folds
        folds_prior = ((f_sum.sum(axis=1, keepdims=True) - f_sum) / (f_count.sum(axis=1, keepdims=True) - f_count)).T
        oof_feats = cp.zeros(data.shape + (n_classes,), dtype=cp.float32)

        self._features = []
        for i in dataset.features:
            for j in range(n_classes):
                self._features.append('{0}_{1}__{2}'.format('multioof', j, i))

        for n in range(data.shape[1]):
            vec = data[:, n]

            # calc folds stats
            enc_dim = int(vec.max() + 1)
            f_sum = cp.zeros((enc_dim, n_classes, n_folds), dtype=cp.float64)
            f_count = cp.zeros((enc_dim, 1, n_folds), dtype=cp.float64)

            scatter_add(f_sum, (vec, target, folds), 1)
            scatter_add(f_count, (vec, 0, folds), 1)

            # calc total stats
            t_sum = f_sum.sum(axis=2, keepdims=True)
            t_count = f_count.sum(axis=2, keepdims=True)

            # calc oof stats
            oof_sum = t_sum - f_sum
            oof_count = t_count - f_count

            # (N x N_classes x 1 + 1 x 1 x N_alphas * N x N_classes x 1) / (N x 1 x 1 + N x 1 x 1) -> N x N_classes x N_alphas
            candidates = ((oof_sum[vec, :, folds, cp.newaxis] + alphas * folds_prior[folds, :, cp.newaxis])
                          / (oof_count[vec, :, folds, cp.newaxis] + alphas)).astype(cp.float32)

            # norm over 1 axis
            candidates /= candidates.sum(axis=1, keepdims=True)

            idx = self.score_func(candidates, target)
            oof_feats[:, n] = candidates[..., idx]
            enc = ((t_sum[..., 0] + alphas[0, 0, idx] * prior) / (t_count[..., 0] + alphas[0, 0, idx])).astype(cp.float32)
            enc /= enc.sum(axis=1, keepdims=True)

            self.encodings.append(enc)
        output = dataset.empty()
        
        oof_feats = cudf.DataFrame(oof_feats.reshape((data.shape[0], -1)), index=dataset.data.index, columns=self.features)
        
        output.set_data(oof_feats, self.features, NumericRole(cp.float32, prob=True))
        return output

    def transform(self, dataset: GpuNumericalDataset) -> GpuNumericalDataset:
        """Transform categorical dataset to target encoding.

        Args:
            dataset: Cudf or Cupy or DaskCudf dataset of categorical features.

        Returns:
            Respective dataset with encoded labels.

        """

        assert isinstance(dataset , GpuNumericalDataset.__args__),\
               'TargetEncoder_gpu works only with CupyDataset, CudfDataset, DaskCudfDataset'

        if isinstance(dataset, DaskCudfDataset):
            return self.transform_daskcudf(dataset)
        else:
            return self.transform_cupy(dataset)

    def transform_daskcudf(self, dataset):
        raise NotImplementedError

    def transform_cupy(self, dataset: GpuNumericalDataset) -> CupyDataset:
        """Transform categorical dataset to target encoding.

        Args:
            dataset: Cudf or Cupy dataset of categorical features.

        Returns:
            Cupy dataset with encoded labels.

        """
        # checks here
        super().transform(dataset)
        # convert to accepted dtype and get attributes
        dataset = dataset.to_cupy()
        data = dataset.data

        # transform
        out = cp.zeros(data.shape + (self.n_classes,), dtype=cp.float32)
        for n, enc in enumerate(self.encodings):
            out[:, n] = enc[data[:, n]]

        out = out.reshape((data.shape[0], -1))
        # create resulted
        output = dataset.empty()
        output.set_data(out, self.features, NumericRole(cp.float32, prob=True))

        return output


class CatIntersections_gpu(LabelEncoder_gpu):
    """Build label encoded intersections of categorical variables."""

    _fit_checks = (categorical_check,)
    _transform_checks = ()
    _fname_prefix = 'inter'

    def __init__(self, subs: Optional[int] = None, random_state: int = 42,
                 intersections: Optional[Sequence[Sequence[str]]] = None, max_depth: int = 2):
        """Create label encoded intersection columns for categories.

        Args:
            intersections: Columns to create intersections.
              Default is None - all.
            max_depth: Max intersection depth.

        """
        super().__init__(subs, random_state)
        self.intersections = intersections
        self.max_depth = max_depth

    @staticmethod
    def _make_category(df: cudf.DataFrame, cols: Sequence[str]) -> cudf.DataFrame:
        """Make hash for category interactions.

        Args:
            df: Input DataFrame
            cols: List of columns

        Returns:
            Hash cudf.DataFrame.

        """

        res = None

        for col in cols:
            if res is None:
                res = df[col].astype('str')
            else:
                res = res + '_' + df[col].astype('str')

        res = res.hash_values()

        #no need to make columns anything meaningfull here, just enumeration is fine
        return cudf.DataFrame(res, columns=['({0})'.format('__'.join(cols))], index=df.index)

    def _build_df(self, dataset: GpuNumericalDataset) -> GpuNumericalDataset:
        """

        Args:
            dataset: Cudf or Cupy or DaskCudf dataset of categorical features.

        Returns:
            Dataset.
        """
        col_names = []
        if type(dataset) == DaskCudfDataset:
            df = dataset.data
            roles = {}
            new_df = []
            
            for comb in self.intersections:
                name = '({0})'.format('__'.join(comb))
                col_names.append(name)
                new_df.append(df.map_partitions(self._make_category, comb))
                roles[name] = CategoryRole(object, unknown=max((dataset.roles[x].unknown for x in comb)),
                                           label_encoded=True)
            for data in new_df:
                mapper = dict(zip(np.arange(len(col_names)), col_names))
            new_df = dask_cudf.concat(new_df, axis=1).rename(columns=mapper).persist()

        else:
            #dataset = dataset.to_cudf()
            df = dataset.data

            roles = {}
            new_df = cudf.DataFrame(index=df.index)
            for comb in self.intersections:
                name = '({0})'.format('__'.join(comb))
                col_names.append(name)
                new_df[name] = self._make_category(df, comb)

                roles[name] = CategoryRole(object, unknown=max((dataset.roles[x].unknown for x in comb)),
                                           label_encoded=True)
        output = dataset.empty()
        output.set_data(new_df, col_names, roles)

        return output

    def fit(self, dataset: GpuNumericalDataset):
        """Create label encoded intersections and save mapping.

        Args:
            dataset: Cudf or Cupy or DaskCudf dataset of categorical features.

        Returns:
            self.

        """
        # set transformer names and add checks
        for check_func in self._fit_checks:
            check_func(dataset)

        if self.intersections is None:
            self.intersections = []
            for i in range(2, min(self.max_depth, len(dataset.features)) + 1):
                self.intersections.extend(list(combinations(dataset.features, i)))

        inter_dataset = self._build_df(dataset)
        return super().fit(inter_dataset)

    def transform(self, dataset: GpuNumericalDataset) -> GpuNumericalDataset:
        """Create label encoded intersections and apply mapping

        Args:
            dataset: Cudf or Cupy or DaskCudf dataset of categorical features

        Returns:

        """
        inter_dataset = self._build_df(dataset)
        return super().transform(inter_dataset)


class OrdinalEncoder_gpu(LabelEncoder_gpu):
    """
    Encoding ordinal categories into numbers.
    Number type categories passed as is,
    object type sorted in ascending lexicographical order.
    """
    _fit_checks = (categorical_check,)
    _transform_checks = ()
    _fname_prefix = 'ord'
    _fillna_val = cp.nan

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._output_role = NumericRole(cp.float32)

    def fit_daskcudf(self, dataset: DaskCudfDataset):
        """Estimate label frequencies and create encoding dicts.

        Args:
            dataset: DaskCudfDataset of categorical features.

        """
        # set transformer names and add checks
        LAMLTransformer.fit(self, dataset)
        # set transformer features
        roles = dataset.roles
        # convert to accepted dtype and get attributes
        self.dicts = {}

        daskcudf_data = dataset.data

        for i in daskcudf_data.columns:
            role = roles[i]
            try:
                flg_number = cp.issubdtype(role.dtype, cp.number)
            except TypeError:
                flg_number = False

            if not flg_number:
                co = role.unknown
                cnts = daskcudf_data[i].value_counts(dropna=False)
                cnts = cnts.astype(np.float32)[cnts > co].compute().reset_index()
                cnts = cudf.Series(cnts['index'].astype(str).rank(), index=cnts['index'], dtype=np.float32)
                cnts = cnts.append(cudf.Series([cnts.shape[0] + 1], index=[cp.nan], dtype=np.float32))
                self.dicts[i] = cnts

        return self

    def fit_cupy(self, dataset: GpuNumericalDataset):
        """Estimate label frequencies and create encoding dicts.

        Args:
            dataset: Cudf or Cupy dataset of categorical features.

        """
        # set transformer names and add checks
        LAMLTransformer.fit(self, dataset)
        # set transformer features

        # convert to accepted dtype and get attributes
        roles = dataset.roles
        subs = self._get_df(dataset)

        self.dicts = {}
        for i in subs.columns:
            role = roles[i]
            try:
                flg_number = cp.issubdtype(role.dtype, cp.number)
            except TypeError:
                flg_number = False

            if not flg_number:
                co = role.unknown
                cnts = subs[i].value_counts(dropna=False)
                cnts = cnts.astype(np.float32)[cnts > co].reset_index()
                cnts = cudf.Series(cnts['index'].astype(str).rank(), index=cnts['index'], dtype=np.float32)
                cnts = cnts.append(cudf.Series([cnts.shape[0] + 1], index=[cp.nan], dtype=np.float32))
                self.dicts[i] = cnts
        return self
