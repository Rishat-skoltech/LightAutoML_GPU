"""Dask_cudf reader."""

from typing import Any, Union, Dict, List, Sequence, TypeVar, Optional

import numpy as np

import cupy as cp

from dask_cudf.core import DataFrame
from dask_cudf.core import Series

from .cudf_reader import CudfReader
from ..dataset.daskcudf_dataset import DaskCudfDataset

from .guess_roles import get_numeric_roles_stat, calc_encoding_rules, rule_based_roles_guess, \
    get_category_roles_stat, calc_category_rules, rule_based_cat_handler_guess
from ..dataset.base import valid_array_attributes, array_attr_roles
from ..dataset.roles import ColumnRole, DropRole, DatetimeRole, CategoryRole, NumericRole
from ..dataset.utils import roles_parser
from ..tasks import Task
from ..utils.logging import get_logger

logger = get_logger(__name__)

# roles, how it's passed to automl
RoleType = TypeVar('RoleType', bound=ColumnRole)
RolesDict = Dict[str, RoleType]

# how user can define roles
UserDefinedRole = Optional[Union[str, RoleType]]

UserDefinedRolesDict = Dict[UserDefinedRole, Sequence[str]]
UserDefinedRolesSequence = Sequence[UserDefinedRole]
UserRolesDefinition = Optional[Union[UserDefinedRole, UserDefinedRolesDict,
                                     UserDefinedRolesSequence]]

#this decorator works bad with a distributed client
#@record_history(enabled=False)
class DaskCudfReader(CudfReader):
    """
    Reader to convert :class:`~dask_cudf.core.DataFrame` to
        AutoML's :class:`~lightautoml.dataset.daskcudf_dataset.DaskCudfDataset`.
    Stages:

        - Drop obviously useless features.
        - Convert roles dict from user format to automl format.
        - Simple role guess for features without input role.
        - Create cv folds.
        - Create initial PandasDataset.
        - Optional: advanced guessing of role and handling types.

    """

    def __init__(self, task: Task, samples: Optional[int] = 100000,
                 max_nan_rate: float = 0.999, max_constant_rate: float = 0.999,
                 cv: int = 5, random_state: int = 42,
                 roles_params: Optional[dict] = None, n_jobs: int = 4,
                 # params for advanced roles guess
                 advanced_roles: bool = True, numeric_unique_rate: float = .999,
                 max_to_3rd_rate: float = 1.1,
                 binning_enc_rate: float = 2, raw_decr_rate: float = 1.1,
                 max_score_rate: float = .2, abs_score_val: float = .04,
                 drop_score_co: float = .01,
                 frac: float = 1.0, compute: bool = True,
                 **kwargs: Any):
        """

        Args:
            task: Task object.
            samples: Not used, kept for the inheritance sake.
            max_nan_rate: Maximum nan-rate.
            max_constant_rate: Maximum constant rate.
            cv: CV Folds.
            random_state: Random seed.
            roles_params: dict of params of features roles. \
                Ex. {'numeric': {'dtype': np.float32}, 'datetime': {'date_format': '%Y-%m-%d'}}
                It's optional and commonly comes from config
            n_jobs: Int number of processes.
            advanced_roles: Param of roles guess (experimental, do not change).
            numeric_unqiue_rate: Param of roles guess (experimental, do not change).
            max_to_3rd_rate: Param of roles guess (experimental, do not change).
            binning_enc_rate: Param of roles guess (experimental, do not change).
            raw_decr_rate: Param of roles guess (experimental, do not change).
            max_score_rate: Param of roles guess (experimental, do not change).
            abs_score_val: Param of roles guess (experimental, do not change).
            drop_score_co: Param of roles guess (experimental, do not change).
            frac: Fraction of the data to sample when checking role type.
            compute: if True reader transfers sample to a signle GPU before guessing roles.
            **kwargs: For now not used.

        """
        self.compute = compute
        self.frac = frac
        super().__init__(task, samples, max_nan_rate, max_constant_rate, cv,
                         random_state, roles_params, n_jobs, advanced_roles,
                         numeric_unique_rate, max_to_3rd_rate, binning_enc_rate,
                         raw_decr_rate, max_score_rate, abs_score_val,
                         drop_score_co, **kwargs)

    def fit_read(self, train_data: DataFrame, features_names: Any = None,
                 roles: UserDefinedRolesDict = None,
                 **kwargs: Any) -> DaskCudfDataset:
        """Get dataset with initial feature selection.

        Args:
            train_data: Input data.
            features_names: Ignored. Just to keep signature.
            roles: Dict of features roles in format
              ``{RoleX: ['feat0', 'feat1', ...], RoleY: 'TARGET', ....}``.
            **kwargs: Can be used for target/group/weights.

        Returns:
            Dataset with selected features.

        """
        logger.info('Train data shape: {}'.format(train_data.shape))

        if roles is None:
            roles = {}
        # transform roles from user format {RoleX: ['feat0', 'feat1', ...], RoleY: 'TARGET', ....}
        # to automl format {'feat0': RoleX, 'feat1': RoleX, 'TARGET': RoleY, ...}
        parsed_roles = roles_parser(roles)
        # transform str role definition to automl ColumnRole
        attrs_dict = dict(zip(array_attr_roles, valid_array_attributes))

        for feat in parsed_roles:
            r = parsed_roles[feat]
            if isinstance(r, str):
                # get default role params if defined
                r = self._get_default_role_from_str(r)

            # check if column is defined like target/group/weight etc ...
            if r.name in attrs_dict:
                # defined in kwargs is rewrited.. TODO: Maybe raise warning if rewrited?
                # TODO: Think, what if multilabel or multitask? Multiple column target ..
                # TODO: Maybe for multilabel/multitask make target only avaliable in kwargs??
                self._used_array_attrs[attrs_dict[r.name]] = feat
                kwargs[attrs_dict[r.name]] = train_data[feat]
                r = DropRole()

            # add new role
            parsed_roles[feat] = r

        assert 'target' in kwargs, 'Target should be defined'
        self.target = kwargs['target'].name
        kwargs['target'] = self._create_target(kwargs['target'])

        # TODO: Check target and task
        # get subsample if it needed
        subsample = train_data
        if self.frac is not None and self.frac < 1.0:
            subsample = subsample.sample(frac = self.frac, random_state=42)
        if self.compute:
            subsample = subsample.compute()
        else:
            subsample = subsample.persist()

        # infer roles
        for feat in subsample.columns:
            assert isinstance(feat, str), 'Feature names must be string,' \
                   ' find feature name: {}, with type: {}'.format(feat, type(feat))
            if feat in parsed_roles:
                r = parsed_roles[feat]
                # handle datetimes
                if r.name == 'Datetime':
                    # try if it's ok to infer date with given params
                    if self.compute:
                        self._try_datetime(subsample[feat], r)
                    else:
                        subsample[feat].map_partitions(self._try_datetime, r,
                                                       meta=(None, None)).compute()
                # replace default category dtype for numeric roles dtype
                #if cat col dtype is numeric
                if r.name == 'Category':
                    # default category role
                    cat_role = self._get_default_role_from_str('category')
                    # check if role with dtypes was exactly defined
                    try:
                        flg_default_params = feat in roles['category']
                    except KeyError:
                        flg_default_params = False

                    if flg_default_params and \
                       not np.issubdtype(cat_role.dtype, np.number) and \
                       np.issubdtype(subsample.dtypes[feat], np.number):
                        r.dtype = self._get_default_role_from_str('numeric').dtype
            else:
                # if no - infer
                is_ok_feature = False
                if self.compute:
                    is_ok_feature = self._is_ok_feature(subsample[feat])
                else:
                    is_ok_feature = subsample[feat].map_partitions(self._is_ok_feature,
                                                    meta=(None, '?')).compute().all()
                if is_ok_feature:
                    r = self._guess_role(subsample[feat])
                else:
                    r = DropRole()

            # set back
            if r.name != 'Drop':
                self._roles[feat] = r
                self._used_features.append(feat)
            else:
                self._dropped_features.append(feat)

        assert len(self.used_features) > 0, 'All features are excluded for some reasons'

        dataset = DaskCudfDataset(data=train_data[self.used_features].persist(),
                                  roles=self.roles, task=self.task, **kwargs)

        return dataset

    def _create_target(self, target: Series):
        """Validate target column and create class mapping is needed

        Args:
            target: Column with target values.

        Returns:
            Transformed target.

        """
        self.class_mapping = None

        if self.task.name != 'reg':
            # expect binary or multiclass here
            cnts = target.value_counts(dropna=False).compute()
            assert not cnts.index.isna().any(), 'Nan in target detected'
            unqiues = cnts.index.values
            srtd = cp.sort(unqiues)
            self._n_classes = len(unqiues)

            # case - target correctly defined and no mapping
            if (cp.arange(srtd.shape[0]) == srtd).all():

                assert srtd.shape[0] > 1, 'Less than 2 unique values in target'
                if self.task.name == 'binary':
                    assert srtd.shape[0] == 2, 'Binary task and more than 2 values in target'
                return target

            # case - create mapping
            self.class_mapping = {n: x for (x, n) in enumerate(cp.asnumpy(unqiues))}
            return target.astype(np.int32).persist()

        assert not target.compute().isna().any(), 'Nan in target detected'
        return target

    def _guess_role(self, feature: Series) -> RoleType:
        """Try to infer role, simple way.

        If convertable to float -> number.
        Else if convertable to datetime -> datetime.
        Else category.

        Args:
            feature: Column from dataset.

        Returns:
            Feature role.

        """
        # TODO: Plans for advanced roles guessing
        # check if default numeric dtype defined
        num_dtype = self._get_default_role_from_str('numeric').dtype
        # check if feature is number
        try:
            if self.compute:
                _ = feature.astype(num_dtype)
            else:
                _ = feature.astype(num_dtype).compute()
            return NumericRole(num_dtype)
        except ValueError:
            pass
        except TypeError:
            pass

        # check if default format is defined
        date_format = self._get_default_role_from_str('datetime').format
        # check if it's datetime
        guessed_role = None
        if self.compute:
            guessed_role = DatetimeRole(np.datetime64, date_format=date_format)\
                           if self._is_datetimable(feature, date_format)\
                           else CategoryRole(object)
        else:
            are_datetime = feature.map_partitions(self._is_datetimable,
                                   date_format, meta=(None, 'U')).compute()
            guessed_role = DatetimeRole(np.datetime64, date_format=date_format)\
                           if are_datetime.all() else CategoryRole(object)
        return guessed_role

    def read(self, data: DataFrame, features_names: Any = None,
             add_array_attrs: bool = False) -> DaskCudfDataset:
        """Read dataset with fitted metadata.

        Args:
            data: Data.
            features_names: Not used.
            add_array_attrs: Additional attributes, like
              target/group/weights/folds.

        Returns:
            Dataset with new columns.

        """

        kwargs = {}
        if add_array_attrs:
            for array_attr in self.used_array_attrs:
                col_name = self.used_array_attrs[array_attr]
                try:
                    val = data[col_name]
                except KeyError:
                    continue

                if array_attr == 'target' and self.class_mapping is not None:
                    val = val.map_partitions(self._apply_class_mapping,
                                             data.index, col_name).persist()
                kwargs[array_attr] = val

        dataset = DaskCudfDataset(data[self.used_features], roles=self.roles,
                                  task=self.task, **kwargs)

        return dataset
