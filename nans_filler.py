import os
import numpy as np
import pandas as pd
import pickle
from catboost import CatBoostClassifier
from catboost import CatBoostRegressor

from preproc_position import position_preproc_by_Igor


UNEMPLOYED = (
    'Студент',
    'Не работаю'
)
CAT_COLUMNS = [
    'education',
    'employment status',
    'Value',
    'Position',
    'Gender',
    'Family status',
    'ChildCount',
    'SNILS',
    'Merch_code',
    'Loan_term',
    'Goods_category',
]
REG_COLUMNS = [
    'MonthProfit',
    'MonthExpense',
    'Loan_amount',
]
cat_imputer = pickle.load(open('./models/simple.cat', 'rb'))
reg_imputer = pickle.load(open('./models/simple.reg', 'rb'))


def unemployed_nansfiller(df: pd.DataFrame) -> pd.DataFrame:
    '''Заполняем пропуски в JobStartDate, Value и Position для безработных'''
    for employment_status in UNEMPLOYED:
        df['Position'][
            (df['employment status'] == employment_status) & 
            (df['JobStartDate'].isna())
        ] = 'Безработный'
        df['Value'][
            (df['employment status'] == employment_status) & 
            (df['JobStartDate'].isna())
        ] = '0 месяцев 0 лет'

    return df


def dates_nansfiller(df: pd.DataFrame) -> pd.DataFrame:
    '''Заполняем пропуски в датах pd.Timestamp.max'''
    df['JobStartDate'].fillna(
        pd.to_datetime(pd.to_datetime(pd.Timestamp.max)), inplace=True)
    df['BirthDate'].fillna(
        pd.to_datetime(pd.to_datetime(pd.Timestamp.max)), inplace=True)
    
    return df


def Yaro_ml_filler(df: pd.DataFrame) -> pd.DataFrame:
    '''Функция заполнения NaNs с помощью предобученных моделей'''
    for col in df.columns:
        nans_frame = df[(df[col].isna())] 
        if nans_frame.shape[0]:
            nans_frame['Position'] = position_preproc_by_Igor(nans_frame)
            nans_frame[CAT_COLUMNS] = cat_imputer.transform(nans_frame[CAT_COLUMNS])
            nans_frame[REG_COLUMNS] = reg_imputer.transform(nans_frame[REG_COLUMNS])
            nans_frame['BirthDate'] = nans_frame['BirthDate'].dt.year
            nans_frame['JobStartDate'] = nans_frame['JobStartDate'].dt.year
            if os.path.exists(f'./models/{col}.cls'):
                model = CatBoostClassifier()
                model.load_model(f'./models/{col}.cls')
            elif os.path.exists(f'./models/{col}.reg'):
                model = CatBoostRegressor()
                model.load_model(f'./models/{col}.reg')
            else:
                continue
            X = nans_frame.drop(col, axis=1)
            pred_y = model.predict(X)
            df.loc[nans_frame.index, col] = pred_y

    return df


def fill_nans_pipe(df: pd.DataFrame) -> pd.DataFrame:
    '''Заполняю пропуски, используя лучшие проверенные стратегии'''
    # Заполняю пропуски для безработных
    df = unemployed_nansfiller(df)
    # Затем пропуски в датах
    df = dates_nansfiller(df)
    # Потом заполняю пропуски с использованием предобученных моделей
    df = Yaro_ml_filler(df)
    # И наконец оставшиеся пропуски с помощью простых стратегий
    # на основе статистик, полученных из исходного датасета,
    # сохраненных в соответствующих объектах SKLearn SimpleImputer
    df[CAT_COLUMNS] = cat_imputer.transform(df[CAT_COLUMNS])
    df[REG_COLUMNS] = reg_imputer.transform(df[REG_COLUMNS])

    return df