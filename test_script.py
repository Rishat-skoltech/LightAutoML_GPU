import os
import pickle
import time

from lightautoml.dataset.np_pd_dataset_cupy import *
from lightautoml.dataset.roles import *
from lightautoml.dataset.utils import roles_parser

from lightautoml.tasks import Task

from lightautoml.transformers import numeric_gpu, categorical_gpu, datetime_gpu

data = pd.read_csv('./example_data/test_data_files/sampled_app_train.csv',
                       usecols=['TARGET', 'NAME_CONTRACT_TYPE', 'AMT_CREDIT',
                                'NAME_TYPE_SUITE', 'AMT_GOODS_PRICE',
                                'DAYS_BIRTH', 'DAYS_EMPLOYED'])

# Fix dates and convert to date type

data['BIRTH_DATE'] = np.datetime64('2018-01-01') + data['DAYS_BIRTH'].astype(np.dtype('timedelta64[D]'))
data['EMP_DATE'] = np.datetime64('2018-01-01') + np.clip(data['DAYS_EMPLOYED'], None, 0).astype(np.dtype('timedelta64[D]'))
data.drop(['DAYS_BIRTH', 'DAYS_EMPLOYED'], axis=1, inplace=True)

# Create folds
data['__fold__'] = np.random.randint(0, 5, len(data))

# Print data head
print(data.head())

# # Set roles for columns

check_roles = {
    TargetRole(): 'TARGET',
    CategoryRole(dtype=str): ['NAME_CONTRACT_TYPE', 'NAME_TYPE_SUITE'],
    NumericRole(np.float32): ['AMT_CREDIT', 'AMT_GOODS_PRICE'],
    DatetimeRole(seasonality=['y', 'm', 'wd']): ['BIRTH_DATE', 'EMP_DATE'],
    FoldsRole(): '__fold__'
}

# create Task
task = Task('binary')

# # Creating PandasDataSet
pd_dataset = PandasDataset(data, roles_parser(check_roles), task=task)

cudf_dataset = pd_dataset.to_cudf()

check_roles = {
    NumericRole(np.float32): ['AMT_CREDIT', 'AMT_GOODS_PRICE'],
}

numeric_dataset = CudfDataset(data=cudf_dataset.data[['AMT_CREDIT',\
                                                      'AMT_GOODS_PRICE']].copy(),
                        roles=roles_parser(check_roles),
                        task=task)
check_roles = {
    DatetimeRole(seasonality=['y', 'm', 'wd']): ['BIRTH_DATE', 'EMP_DATE'],
}

datetime_dataset = CudfDataset(data=cudf_dataset.data[['BIRTH_DATE',\
                                                      'EMP_DATE']].copy(),
                        roles=roles_parser(check_roles),
                        task=task)


filler = numeric_gpu.FillnaMedian()
filled_dataset = filler.fit_transform(numeric_dataset)

date_seasons = datetime_gpu.DateSeasons()
date_dataset = date_seasons.fit_transform(datetime_dataset)

from lightautoml.ml_algo import linear_gpu_bak

check_roles = {
    TargetRole(): 'TARGET',
}

# create Task
task = Task('binary')

# # Creating PandasDataSet
target_dataset = CudfDataset(data[['TARGET']], roles_parser(check_roles), task=task)


filled_dataset = filled_dataset.to_cudf()
filled_dataset = filled_dataset.concat([filled_dataset, date_dataset.to_cudf(), target_dataset])

check_roles = {
    TargetRole(cp.float32): 'TARGET',
    NumericRole(cp.float32): ['fillnamed__AMT_CREDIT', 'fillnamed__AMT_GOODS_PRICE',
       'season_y__BIRTH_DATE', 'season_m__BIRTH_DATE', 'season_wd__BIRTH_DATE',
       'season_y__EMP_DATE', 'season_m__EMP_DATE', 'season_wd__EMP_DATE'],
}

task = Task('binary', device='gpu')

full_dataset = CupyDataset(filled_dataset.data.values,
                            features=filled_dataset.data.columns.to_list(),
                              roles=roles_parser(check_roles),
                              task=task,
                              **{'target': filled_dataset.data['TARGET']})

from lightautoml.validation.utils import create_validation_iterator

train_valid = create_validation_iterator(full_dataset[:9000], full_dataset[9000:], n_folds=10)

linear_one = linear_gpu_bak.LinearLBFGS()
linear_one.fit_predict(train_valid)




