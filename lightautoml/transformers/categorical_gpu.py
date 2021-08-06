"""Categorical features transformers."""

from itertools import combinations
from typing import Optional, Union, List, Sequence, cast

import cupy as cp
from cupyx import scatter_add
import cudf

from cuml.preprocessing import OneHotEncoder as one_hot

from .base import LAMLTransformerGPU
from ..dataset.base import LAMLDataset
from ..dataset.np_pd_dataset_cupy import CudfDataset, CupyDataset, DaskCudfDataset, CSRSparseDataset
from ..dataset.roles import CategoryRole, NumericRole

# type - something that can be converted to cudf dataset
NumericalDataset = Union[CupyDataset, CudfDataset, DaskCudfDataset]
NumericalOrSparse = Union[CupyDataset, CSRSparseDataset]

def in_gpu(dataset):
    return (type(dataset) in [CupyDataset, CudfDataset, DaskCudfDataset])

def categorical_check(dataset: LAMLDataset):
    """Check if all passed vars are categories.

    Raises AssertionError if non-categorical features are present.

    Args:
        dataset: LAMLDataset to check.

    """
    roles = dataset.roles
    features = dataset.features
    for f in features:
        assert roles[f].name == 'Category', 'Only categories accepted in this transformer'


def oof_task_check(dataset: LAMLDataset):
    """Check if all passed vars are categories.

    Args:
        dataset: Input.

    """
    task = dataset.task
    assert task.name in ['binary', 'reg'], 'Only binary and regression tasks supported in this transformer'


def multiclass_task_check(dataset: LAMLDataset):
    """Check if all passed vars are categories.

    Args:
        dataset: Input.

    Returns:

    """
    task = dataset.task
    assert task.name in ['multiclass'], 'Only multiclass tasks supported in this transformer'


def encoding_check(dataset: LAMLDataset):
    """Check if all passed vars are categories.

    Args:
        dataset: Input.

    """
    roles = dataset.roles
    features = dataset.features
    for f in features:
        assert roles[
            f].label_encoded, 'Transformer should be applied to category only after label encoding. Feat {0} is {1}'.format(
            f, roles[f])


class LabelEncoder(LAMLTransformerGPU):
    """Simple LabelEncoder in order of frequency.

    Labels are integers from 1 to n. Unknown category encoded as 0.
    NaN is handled as a category value.

    """
    _fit_checks = (categorical_check,)
    _transform_checks = ()
    _fname_prefix = 'le'

    # _output_role = CategoryRole(np.int32, label_encoded=True)
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

    def _get_df(self, dataset: NumericalDataset) -> cudf.DataFrame:
        """Get df and sample.

        Args:
            dataset: Input dataset.

        Returns:
            Subsample.

        """
        dataset = dataset.to_cudf()
        df = dataset.data

        if self.subs is not None and df.shape[0] >= self.subs:
            subs = df.sample(n=self.subs, random_state=self.random_state)
        else:
            subs = df

        return subs

    def fit(self, dataset: NumericalDataset):
        """Estimate label frequencies and create encoding dicts.

        Args:
            dataset: Cudf or Cupy dataset of categorical features.

        Returns:
            self.

        """
        # set transformer names and add checks
        super().fit(dataset)
        # set transformer features

        # convert to accepted dtype and get attributes
        roles = dataset.roles
        subs = self._get_df(dataset)

        self.dicts = {}
        for i in subs.columns:
            role = roles[i]
            # TODO: think what to do with this warning
            co = role.unknown

            cnts = subs[i].value_counts(dropna=False).reset_index() \
                .sort_values([i, 'index'], ascending=[False, True])

            vals = cnts[cnts[i] > co]['index']
            self.dicts[i] = cudf.Series(cp.arange(vals.shape[0], dtype=cp.int32) + 1, index=vals)

        return self

    def transform(self, dataset: NumericalDataset) -> CupyDataset:
        """Transform categorical dataset to int labels.

        Args:
            dataset: Cudf or Cupy dataset of categorical features.

        Returns:
            Cupy dataset with encoded labels.

        """
        # checks here
        super().transform(dataset)
        # convert to accepted dtype and get attributes
        dataset = dataset.to_cudf()
        df = dataset.data

        # transform
        new_arr = cp.empty(dataset.shape, dtype=self._output_role.dtype)

        for n, i in enumerate(df.columns):
            # to be compatible with OrdinalEncoder
            if i in self.dicts:
                new_arr[:, n] = df[i].map(self.dicts[i]).fillna(self._fillna_val).values
            else:
                new_arr[:, n] = df[i].values.astype(self._output_role.dtype)

        # create resulted
        output = dataset.empty().to_cupy()
        output.set_data(new_arr, self.features, self._output_role)

        return output

class OHEEncoder(LAMLTransformerGPU):
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

    def __init__(self, make_sparse: Optional[bool] = False, total_feats_cnt: Optional[int] = None, dtype: type = cp.float32):
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
            assert self.total_feats_cnt is not None, 'Param total_feats_cnt should be defined if make_sparse is None'

    def fit(self, dataset: NumericalDataset):
        """Calc output shapes.

        Automatically do ohe in sparse form if approximate fill_rate < `0.2`.

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
        min_idx = cp.asnumpy(data.min(axis=0))

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

    def transform(self, dataset: NumericalDataset) -> NumericalOrSparse:
        """Transform categorical dataset to ohe.

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
        data = self.ohe.transform(data)

        # create resulted
        output = dataset.empty()
        if self.make_sparse:
            output = output.to_csr()

        output.set_data(data, self.features, NumericRole(self.dtype))

        return output

class FreqEncoder(LAMLTransformerGPU):
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

        self._output_role = NumericRole(cp.float32)

    def fit(self, dataset: NumericalDataset):
        """Estimate label frequencies and create encoding dicts.

        Args:
            dataset: Cudf or Cupy dataset of categorical features

        Returns:
            self.
            
        """
        # set transformer names and add checks
        super().fit(dataset)
        # set transformer features

        # convert to accepted dtype and get attributes
        dataset = dataset.to_cudf()
        df = dataset.data

        self.dicts = {}
        for i in df.columns:
            # we make assertion in checks, so cast is ok
            # TODO: think what to do with this warning
            cnts = df[i].value_counts(dropna=False)
            self.dicts[i] = cnts[cnts > 1]

        return self

    def transform(self, dataset: NumericalDataset) -> CupyDataset:
        """Transform categorical dataset to int labels.

        Args:
            dataset: Cudf or Cupy dataset of categorical features.

        Returns:
            Numpy dataset with encoded labels.

        """
        # checks here
        super().transform(dataset)
        # convert to accepted dtype and get attributes
        dataset = dataset.to_cudf()
        df = dataset.data

        # transform
        new_arr = cp.empty(dataset.shape, dtype=self._output_role.dtype)

        for n, i in enumerate(df.columns):
            # to be compatible with OrdinalEncoder
            if i in self.dicts:
                new_arr[:, n] = df[i].map(self.dicts[i]).fillna(self._fillna_val).values
            else:
                new_arr[:, n] = df[i].values.astype(self._output_role.dtype)

        # create resulted
        output = dataset.empty().to_cupy()
        output.set_data(new_arr, self.features, self._output_role)

        return output

class TargetEncoder(LAMLTransformerGPU):
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

    def fit(self, dataset: NumericalDataset):
        super().fit_transform(dataset)

    def fit_transform(self, dataset: NumericalDataset) -> CupyDataset:
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
        dataset = dataset.to_cupy()
        data = dataset.data

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

        # TEST THIS PART
        #f_sum += cp.bincount(folds, target, minlength=f_sum.size)
        #f_count += cp.bincount(folds, minlength=f_count.size)
        scatter_add(f_sum, folds, target)
        scatter_add(f_count, folds, 1)

        folds_prior = (f_sum.sum() - f_sum) / (f_count.sum() - f_count)
        oof_feats = cp.zeros(data.shape, dtype=cp.float32)

        for n in range(data.shape[1]):
            vec = data[:, n]

            # calc folds stats
            enc_dim = int(vec.max() + 1)
            f_sum = cp.zeros((enc_dim, n_folds), dtype=cp.float64)
            f_count = cp.zeros((enc_dim, n_folds), dtype=cp.float64)

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

        output = dataset.empty()
        self.output_role = NumericRole(cp.float32, prob=output.task.name == 'binary')
        output.set_data(oof_feats, self.features, self.output_role)

        return output

    def transform(self, dataset: NumericalDataset) -> NumericalOrSparse:
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
        out = cp.zeros(data.shape, dtype=cp.float32)
        for n, enc in enumerate(self.encodings):
            out[:, n] = enc[data[:, n]]

        # create resulted
        output = dataset.empty()
        output.set_data(out, self.features, self.output_role)

        return output


class MultiClassTargetEncoder(LAMLTransformerGPU):
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

    def fit_transform(self, dataset: NumericalDataset) -> CupyDataset:
        """Estimate label frequencies and create encoding dicts.

        Args:
            dataset: Cudf or Cupy dataset of categorical label encoded features.

        Returns:
            CupyDataset - target encoded features.
            
        """
        # set transformer names and add checks
        for check_func in self._fit_checks:
            check_func(dataset)
        # set transformer features

        # convert to accepted dtype and get attributes
        dataset = dataset.to_cupy()
        data = dataset.data
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
        output.set_data(oof_feats.reshape((data.shape[0], -1)), self.features, NumericRole(cp.float32, prob=True))

        return output

    def transform(self, dataset: NumericalDataset) -> NumericalOrSparse:
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


class CatIntersectstions(LabelEncoder):
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
    def _make_category(df: cudf.DataFrame, cols: Sequence[str]) -> cp.ndarray:
        """Make hash for category interactions.

        Args:
            df: Input DataFrame
            cols: List of columns

        Returns:
            Hash cp.ndarray.

        """

        delim = cudf.Series(["_" for _ in range(len(df))])
        res = None

        for col in cols:
            if res is None:
                res = df[col].astype('str')
            else:
                res = res + delim + df[col].astype('str')

        res = res.hash_values()

        return res

    def _build_df(self, dataset: NumericalDataset) -> CudfDataset:
        """

        Args:
            dataset: Cudf or Cupy dataset of categorical features.

        Returns:
            Dataset.
        """
        dataset = dataset.to_cudf()
        df = dataset.data

        roles = {}
        self.res_tensor = {}
        new_df = cudf.DataFrame(index=df.index)
        for comb in self.intersections:
            name = '({0})'.format('__'.join(comb))
            new_df[name] = self._make_category(df, comb)

            roles[name] = CategoryRole(object, unknown=max((dataset.roles[x].unknown for x in comb)), label_encoded=True)

        output = dataset.empty()
        output.set_data(new_df, new_df.columns, roles)
        self.out_tensor = output
        return output

    def fit(self, dataset: NumericalDataset):
        """Create label encoded intersections and save mapping.

        Args:
            dataset: Cudf or Cupy dataset of categorical features.

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

    def transform(self, dataset: NumericalDataset) -> CupyDataset:
        """Create label encoded intersections and apply mapping

        Args:
            dataset: Cudf or Cupy dataset of categorical features

        Returns:

        """
        inter_dataset = self._build_df(dataset)
        return super().transform(inter_dataset)

class OrdinalEncoder(LabelEncoder):
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

    def fit(self, dataset: NumericalDataset):
        """Estimate label frequencies and create encoding dicts.

        Args:
            dataset: Cudf or Cupy dataset of categorical features.

        """
        # set transformer names and add checks
        LAMLTransformerGPU.fit(self, dataset)
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
                cnts = subs[i].value_counts(dropna=True)
                cnts = cnts[cnts > co].reset_index()
                cnts = cudf.Series(cnts['index'].astype(str).rank().values, index=cnts['index'])
                cnts = cnts.append(cudf.Series([cnts.shape[0] + 1], index=[cp.nan]))
                self.dicts[i] = cnts

        return self