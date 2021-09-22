"""Roles guess on gpu."""

from typing import Optional, Union, Dict, Tuple, Any, List

from time import perf_counter
from joblib import Parallel, delayed
from copy import copy
import cudf
import numpy as np
import cupy as cp
import pandas as pd
from lightautoml.dataset.cp_cudf_dataset import CudfDataset, CupyDataset
from lightautoml.dataset.roles import CategoryRole, ColumnRole
from lightautoml.reader.utils_gpu import set_sklearn_folds_gpu
from lightautoml.transformers.base import ChangeRoles
from lightautoml.transformers.base import SequentialTransformer, LAMLTransformer
from lightautoml.transformers.categorical_gpu import TargetEncoder_gpu,\
                         MultiClassTargetEncoder_gpu, LabelEncoder_gpu,\
                         FreqEncoder_gpu, OrdinalEncoder_gpu
from lightautoml.transformers.numeric_gpu import QuantileBinning_gpu

RolesDict = Dict[str, ColumnRole]
Encoder_gpu = Union[TargetEncoder_gpu, MultiClassTargetEncoder_gpu]
GpuFrame = Union[cudf.DataFrame]
GpuDataset = Union[CudfDataset, CupyDataset]

def ginic_gpu(actual: GpuFrame, pred: GpuFrame) -> float:
    """Denormalized gini calculation.

    Args:
        actual_pred: array with true and predicted values
        inds: list of indices for true and predicted values

    Returns:
        Metric value

    """
    gini_sum = 0
    n = actual.shape[0]
    a_s = actual[cp.argsort(pred)]
    a_c = a_s.cumsum()
    gini_sum = a_c.sum() / a_s.sum() - (n + 1) / 2.0
    return gini_sum / n

def gini_normalizedc_gpu(a: GpuFrame, p: GpuFrame) -> float:
    """Calculated normalized gini.

    Args:
        a_p: array with true and predicted values

    Returns:
        Metric value.

    """
    out = ginic_gpu(a, p) / ginic_gpu(a, a)

    assert not np.isnan(out), 'gini index is givin nan, is that ok? {0} and {1}'.format(a, p)
    return out


def gini_normalized_gpu(y: GpuFrame, target: GpuFrame,
                 empty_slice: GpuFrame = None) -> float:
    """Calculate normalized gini index for dataframe data.

    Args:
        y: data.
        true_cols: columns with true data.
        pred_cols: columns with predict data.
        empty_slice: Mask.

    Returns:
        Gini value.

    """
    if empty_slice is None:
        empty_slice = y.isna().values
    all_true = empty_slice.all()
    if all_true:
        return 0.0

    sl = ~empty_slice
    #ndims?
    outp_size = 1 if len(target.shape) <= 1 else target.shape[1]
    pred_size = 1 if len(y.shape) <= 1 else y.shape[1]
    assert pred_size in (1, outp_size),\
           'Shape missmatch. Only calculate NxM vs NxM or Nx1 vs NxM'

    ginis = np.zeros((outp_size,), dtype=np.float32)
    for i in range(outp_size):
        j = min(i, pred_size - 1)
        yp = None
        if pred_size == 1:
            yp = y[sl]
        else:
            yp = y[:, j][sl]
        yt = None
        if outp_size == 1:
            yt = target[sl]
        else:
            yt = target[:, i][sl]

        ginis[i] = gini_normalizedc_gpu(yt, yp)
    return np.abs(ginis).mean()


def get_target_and_encoder_gpu(train: GpuDataset) -> Tuple[Any, type]:
    """Get target encoder and target based on dataset.

    Args:
        train: Dataset.

    Returns:
        (Target values, Target encoder).

    """
    def encode_target(inp, n_out, target_name):
        outp = (inp.values[:, cp.newaxis] == cp.arange(n_out)[cp.newaxis, :])
        #cols = []
        #for i in range(outp.shape[1]):
        #    cols.append(target_name+'_'+str(i))
        return cudf.DataFrame(outp.reshape((outp.shape[0], outp.shape[-1])),
                              index=inp.index)

    target = train.target
    target_name = train.target.name
    if train.task.name == 'multiclass':
        if isinstance(train.data, cudf.DataFrame):
            n_out = cp.max(target.values)+1
            target = encode_target(target, n_out, target_name)
        elif isinstance(train.data, cp.ndarray):
            n_out = cp.max(target)+1
            target = (target[:, cp.newaxis] == cp.arange(n_out)[cp.newaxis, :])
        else:
            raise NotImplementedError

        encoder = MultiClassTargetEncoder_gpu
    else:
        encoder = TargetEncoder_gpu
    return target, encoder


def calc_ginis_gpu(data: Union[GpuFrame, cp.ndarray],
                 target: Union[GpuFrame, cp.ndarray],
                 empty_slice: Union[GpuFrame, cp.ndarray] = None) -> np.ndarray:
    """

    Args:
        data: cp.ndarray or gpu DataFrame.
        target: cp.ndarray or gpu DataFrame.
        empty_slice: cp.ndarray or gpu DataFrame.

    Returns:
        gini.

    """
    orig_len = len(empty_slice.columns)
    new_len = len(data.columns)
    scores = np.zeros(new_len)
    len_ratio = int(new_len/orig_len)
    data = data.values.astype(cp.float32)
    target = target.values.astype(cp.float32)
    empty_slice = empty_slice.values
    for i in range(new_len):
        sl = None
        if empty_slice is not None:
            ind = int(i/len_ratio)
            sl = empty_slice[:, ind]
        scores[i] = gini_normalized_gpu(data[:,i], target,
                                        empty_slice=sl)
    if len_ratio!=1:
        
        scores = scores.reshape((orig_len, len_ratio))
        scores = scores.mean(axis=1)
    return scores


def _get_score_from_pipe_gpu(train: GpuDataset, target: GpuDataset,
      pipe: Optional[LAMLTransformer] = None,
      empty_slice: Optional[Union[GpuFrame, cp.ndarray]] = None) -> np.ndarray:
    """Get normalized gini index from pipeline.

    Args:
        train: gpu Dataset.
        target: gpu Dataset.
        pipe: LAMLTransformer.
        empty_slice: cp.ndarray or gpu DataFrame.

    Returns:
        np.ndarray.

    """
    shape = train.shape
    if pipe is not None:
        train = pipe.fit_transform(train)

    data = copy(train.data)

    scores = calc_ginis_gpu(data, target, empty_slice)
    #print(scores)
    return scores


def get_score_from_pipe_gpu(train: GpuDataset, target: GpuDataset,
         pipe: Optional[LAMLTransformer] = None,
         empty_slice: Optional[GpuFrame] = None, n_jobs: int = 1) -> np.ndarray:
    """Get normalized gini index from pipeline.

    Args:
        train: gpu Dataset.
        target: gpu Dataset.
        pipe: LAMLTransformer.
        empty_slice: gpu DataFrame.
        n_jobs: int

    Returns:
        np.ndarray.

    """
    if n_jobs == 1:
        return _get_score_from_pipe_gpu(train, target, pipe, empty_slice)

    idx = np.array_split(np.arange(len(train.features)), n_jobs)
    idx = [x for x in idx if len(x) > 0]
    n_jobs = len(idx)

    names = [[train.features[x] for x in y] for y in idx]

    with Parallel(n_jobs=n_jobs, prefer='processes', backend='loky', max_nbytes=None) as p:
        res = p(
            delayed(_get_score_from_pipe_gpu)(train[:, name], target, pipe,
                                           empty_slice[name]) for name in names)

    return np.concatenate(list(map(np.array, res)))


def get_numeric_roles_stat_gpu(train: GpuDataset,
                  subsample: Optional[Union[float, int]] = 100000,
               random_state: int = 42, manual_roles: Optional[RolesDict] = None,
               n_jobs: int = 1) -> pd.DataFrame:
    """Calculate statistics about different encodings performances.

    We need it to calculate rules about advanced roles guessing.
    Only for numeric data.

    Args:
        train: Dataset.
        subsample: size of subsample.
        random_state: int.
        manual_roles: Dict.
        n_jobs: int.

    Returns:
        DataFrame.

    """
    if manual_roles is None:
        manual_roles = {}

    roles_to_identify = []
    flg_manual_set = []
    # check for train dtypes
    for f in train.features:
        role = train.roles[f]
        if role.name == 'Numeric':# and f != train.target.name:
            roles_to_identify.append(f)
            flg_manual_set.append(f in manual_roles)

    res = pd.DataFrame(columns=['flg_manual', 'unique', 'unique_rate',\
                   'top_freq_values', 'raw_scores', 'binned_scores',\
                   'encoded_scores','freq_scores', 'nan_rate'],
                   index=roles_to_identify)

    res['flg_manual'] = flg_manual_set
    if len(roles_to_identify) == 0:
        return res

    train = train[:, roles_to_identify]
    train_len = train.shape[0]

    if train.folds is None:
        train.folds = set_sklearn_folds_gpu(train.task, train.target, cv=5,
                                   random_state=random_state, group=train.group)
    if subsample is not None and subsample < train_len:
        #here need to do the remapping
        #train.data = train.data.sample(subsample, axis=0,
        #                               random_state=random_state)
        idx = np.random.RandomState(random_state).permutation(train_len)[:subsample]
        train = train[idx]
        train_len = subsample

    target, encoder = get_target_and_encoder_gpu(train)

    empty_slice = train.data.isna()
    # check scores as is
    res['raw_scores'] = get_score_from_pipe_gpu(train, target,
                                         empty_slice=empty_slice, n_jobs=n_jobs)
    # check unique values
    unique_values = None
    top_freq_values = None
    if isinstance(train.data, cudf.DataFrame):
        #unique_values = [train.data[x].dropna().value_counts() for x in train.data.columns]
        #top_freq_values = [x.max() for x in unique_values]
        #unique_values = [x.shape[0] for x in unique_values]
        desc = train.data.nans_to_nulls().astype(object).describe(include='all')
        unique_values = desc.loc['unique'].astype(np.int32).values[0].get()
        top_freq_values = desc.loc['freq'].astype(np.int32).values[0].get()
    else:
        raise NotImplementedError
    res['unique'] = unique_values
    res['top_freq_values'] = top_freq_values
    res['unique_rate'] = res['unique'] / train_len
    # check binned categorical score
    trf = SequentialTransformer([QuantileBinning_gpu(), encoder()])
    res['binned_scores'] = get_score_from_pipe_gpu(train, target, pipe=trf,
                                                   empty_slice=empty_slice, n_jobs=n_jobs)
    # check label encoded scores
    trf = SequentialTransformer([ChangeRoles(CategoryRole(np.float32)),
                                 LabelEncoder_gpu(), encoder()])
    res['encoded_scores'] = get_score_from_pipe_gpu(train, target, pipe=trf,
                                                   empty_slice=empty_slice, n_jobs=n_jobs)
    # check frequency encoding
    trf = SequentialTransformer([ChangeRoles(CategoryRole(np.float32)), FreqEncoder_gpu()])
    res['freq_scores'] = get_score_from_pipe_gpu(train, target, pipe=trf,
                                                   empty_slice=empty_slice, n_jobs=n_jobs)

    if isinstance(empty_slice, cudf.DataFrame):
        res['nan_rate'] = empty_slice.mean(axis=0).values_host
    else:
        raise NotImplementedError
    #print(res[['raw_scores', 'binned_scores', 'encoded_scores', 'freq_scores']])
    return res


def get_category_roles_stat_gpu(train: GpuDataset,
                            subsample: Optional[Union[float, int]] = 100000,
                            random_state: int = 42,
                            n_jobs: int = 1) -> pd.DataFrame:
    """Search for optimal processing of categorical values.

    Categorical means defined by user or object types.

    Args:
        train: Dataset.
        subsample: size of subsample.
        random_state: seed of random numbers generator.
        n_jobs: number of jobs.

    Returns:
        DataFrame.

    """

    roles_to_identify = []

    dtypes = []

    # check for train dtypes
    for f in train.features:
        role = train.roles[f]
        if role.name == 'Category' and role.encoding_type == 'auto':
            roles_to_identify.append(f)
            dtypes.append(role.dtype)

    res = pd.DataFrame(columns=['unique', 'top_freq_values', 'dtype', 'encoded_scores',
                         'freq_scores'], index=roles_to_identify)

    res['dtype'] = dtypes

    if len(roles_to_identify) == 0:
        return res

    train = train[:, roles_to_identify]
    train_len = train.shape[0]

    if train.folds is None:
        train.folds = set_sklearn_folds_gpu(train.task, train.target, cv=5,
                                            random_state=random_state, group=train.group)
    if subsample is not None and subsample < train_len:
        idx = np.random.RandomState(random_state).permutation(train_len)[:subsample]
        train = train[idx]
        #train.data = train.data.sample(subsample, axis=0, random_state=random_state)
        train_len = subsample

    target, encoder = get_target_and_encoder_gpu(train)
    empty_slice = train.data.isna()

    # check label encoded scores
    trf = SequentialTransformer([LabelEncoder_gpu(), encoder()])
    res['encoded_scores'] = get_score_from_pipe_gpu(train, target, pipe=trf,
                                                  empty_slice=empty_slice, n_jobs=n_jobs)
    # check frequency encoding
    trf = FreqEncoder_gpu()
    res['freq_scores'] = get_score_from_pipe_gpu(train, target, pipe=trf,
                                                  empty_slice=empty_slice, n_jobs=n_jobs)
    # check ordinal encoding
    trf = OrdinalEncoder_gpu()
    res['ord_scores'] = get_score_from_pipe_gpu(train, target, pipe=trf,
                                                  empty_slice=empty_slice, n_jobs=n_jobs)
    #print(res[['encoded_scores', 'freq_scores', 'ord_scores']])
    return res


def get_null_scores_gpu(train: GpuDataset, feats: Optional[List[str]] = None,
                    subsample: Optional[Union[float, int]] = 100000,
                    random_state: int = 42) -> pd.Series:
    """Get null scores.

    Args:
        train: Dataset
        feats: list of features.
        subsample: size of subsample.
        random_state: seed of random numbers generator.

    Returns:
        Series.

    """
    if feats is not None:
        train = train[:, feats]

    shape = train.shape

    if subsample is not None and subsample < shape[0]:
        idx = np.random.RandomState(random_state).permutation(shape[0])[:subsample]
        train = train[idx]
        #train.data = train.data.sample(subsample, axis=0,
        #                               random_state=random_state)

    # check task specific
    target, _ = get_target_and_encoder_gpu(train)

    empty_slice = train.data.isnull()
    notnan = empty_slice.sum(axis=0)
    notnan = (notnan != shape[0]) & (notnan != 0)

    notnan_inds = empty_slice.columns[notnan.values_host]
    empty_slice = empty_slice[notnan_inds]

    scores = np.zeros(shape[1])

    if len(notnan_inds) != 0:
        notnan_inds = np.array(notnan_inds).reshape(-1, 1)
        scores_ = calc_ginis_gpu(empty_slice, target, empty_slice)
        scores[notnan.values_host] = scores_

    res = pd.Series(scores, index=train.features, name='max_score_null')
    #print(res)
    return res
