import sys
import os

import datetime

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_dtype

from sklearn.impute import SimpleImputer

from catboost import CatBoostClassifier
from catboost import CatBoostRegressor

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from pos_prepr_by_Igor import CarierLevel_feature_creator


'''
В модуле реализован конвейер заполнения пропусков во входном фрейме данных
для каждого из признаков в отдельности (кроме, разумеется, целевой переменной).
Заполнение пропусков осуществляется с помощью предобученных
моделей CatBoost. Обучение моделей производилось на исходном датасете.
Основная идея заключается в том, что некоторые исходные признаки имеют
тесную корреляцию, иногда зависимость нелинейная и множественная, поэтому
пропущенное значение можно восстанавливать с помощью моделей точнее,
чем при использовании простых стратегий заполнения. Те параметры, для
которых отсутствуют обученные модели, заполнение производится простыми 
стратегиями (наиболее частое значение для категориальных и среднее для 
числовых признаков).

Может вызываться либо из другого модуля (при этом нужно обращаться к
функции preproc_pipe(df) и передавать заранее загруженный датафрейм),
либо через командную строку с указанием в виде первого аргумента путь 
к *.csv файлу с датафреймом. В первом случае обработанный датафрейм будет 
возвращен при вызове функции, во втором сохранен в виде *_edited.csv
'''


def cat_num_split(df: pd.DataFrame) -> tuple:
    '''Ищем категориальные и числовые признаки в датафрейме'''
    cat_columns = []
    num_columns = []
    for column_name in df.columns:
        if (df[column_name].dtypes == object):
            cat_columns += [column_name]
        else:
            num_columns += [column_name]

    return cat_columns, num_columns


def target_columns_dropper(df: pd.DataFrame) -> pd.DataFrame:
    '''Сбрасываем столбцы с целевыми переменными ответа банков'''
    df = df.copy()
    banks = ['BankA_decision',
             'BankB_decision',
             'BankC_decision',
             'BankD_decision',
             'BankE_decision']
    try:
        df.drop(banks, axis='columns', inplace=True)
    except:
        print('Dataframe have not target columns')

    return df


def create_simple_imputers(df: pd.DataFrame) -> None:
    '''Обучаем простые заполнятели'''
    global cat_imputer
    global num_imputer
    cat_columns, num_columns = cat_num_split(df)
    cat_imputer = SimpleImputer(
        strategy='most_frequent').fit(df[cat_columns])
    num_imputer = SimpleImputer(
        strategy='mean').fit(df[num_columns])


def simple_imputers_filler(df: pd.DataFrame) -> pd.DataFrame:
    '''Заполняем простыми заполнятелями'''
    df = df.copy()
    cat_columns, num_columns = cat_num_split(df)
    df[cat_columns] = cat_imputer.transform(df[cat_columns])
    df[num_columns] = num_imputer.transform(df[num_columns])

    return df


def Yaro_ml_filler(df: pd.DataFrame) -> pd.DataFrame:
    '''Функция заполнения NaNs с помощью предобученных моделей'''
    for col in df.columns:
        nans_frame = df[(df[col].isna())] 
        if nans_frame.shape[0]:
            nans_frame = simple_imputers_filler(nans_frame)
            if os.path.exists(f'./Models/{col}.cls'):
                model = CatBoostClassifier()
                model.load_model(f'./Models/{col}.cls')
            elif os.path.exists(f'./Models/{col}.reg'):
                model = CatBoostRegressor()
                model.load_model(f'./Models/{col}.reg')
            else:
                continue
            X = nans_frame.drop(col, axis=1)
            pred_y = model.predict(X)
            df.loc[nans_frame.index, col] = pred_y
    create_simple_imputers(df)
    df = simple_imputers_filler(df)

    return df


def preproc_pipe(df: pd.DataFrame) -> pd.DataFrame:
    '''Конвейер NaNозаполнятеля'''
    df = df.copy()
    df = df.dropna(how='all')
    df = target_columns_dropper(df)
    df['Position'] = CarierLevel_feature_creator(df)
    # Вот в этом блоке запрятан вооооо-ооот такенный костыль
    # потому что я устал перегонять даты из одного
    # формата в другой и ловить кучу ошибок на этом.
    # Потом нужно будет переделать.
    # -------------------------------------------------------------
    df['JobStartDate'].fillna(pd.to_datetime(datetime.date.today()), inplace=True)
    df['BirthDate'].fillna(df['BirthDate'].median(), inplace=True)
    fd = df.copy()
    df['JobStartDate'] = df['JobStartDate'].dt.year
    df['BirthDate'] = df['BirthDate'].dt.year
    # -------------------------------------------------------------
    create_simple_imputers(df)
    df = Yaro_ml_filler(df)
    df[['BirthDate', 'JobStartDate']] = fd[['BirthDate', 'JobStartDate']]

    return df


if __name__ == "__main__":
    try:
        dataframe_path = sys.argv[1]
    except:
        raise ValueError ('Please use correct parameters: [1]-dataframe path')

    df = pd.read_csv(
    filepath_or_buffer=dataframe_path,
    sep=';',
    index_col='SkillFactory_Id',
    parse_dates=['BirthDate', 'JobStartDate'])

    #print(df.info())
    df = preproc_pipe(df)
    #print(df.info())
    #print(df.head(5))

    path, filename = os.path.split(dataframe_path)
    filename = 'edited_' + filename
    path = path + filename
    df.to_csv(path, sep=';', index=True)