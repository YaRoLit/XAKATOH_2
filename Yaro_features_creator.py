import pandas as pd
from datetime import date

from preproc_position import position_preproc_by_Igor


#==============================================================================
# Новые фичи от Ярослава
#==============================================================================
def Age_feature_creator(df: pd.DataFrame) -> pd.Series:
    '''Создаем признак "возраст"'''
    df = df.copy()

    return date.today().year - df['BirthDate'].dt.year


def Num_value_feature_creator(df: pd.DataFrame) -> pd.Series:
    '''Создаем признак "стаж работы" из даты начала работы'''
    df = df.copy()

    return date.today().year - df['JobStartDate'].dt.year


def Monthly_credit_payment(df: pd.DataFrame) -> pd.Series:
    '''Создаем функцию расчета ежемесячного платёжа'''
    df = df.copy()

    return df['Loan_amount'] / df['Loan_term']


def Payment_to_income(df: pd.DataFrame) -> pd.Series:
    '''Создаем признак "показатель долговой нагрузки"'''
    df = df.copy()
    monthly_payment = Monthly_credit_payment(df) + df['MonthExpense']

    return monthly_payment / df['MonthProfit']


#==============================================================================
# Пайплайн для сбора новых фич для используемой модели
#==============================================================================
def Yaro_features_creator_pipe(df: pd.DataFrame) -> pd.DataFrame:
   '''Пайплайн  создания признаков, необходимых для работы модели'''
   df = df.copy()
   # Преобразование столбца Position
   df['Position'] = position_preproc_by_Igor(df)
   # Вводим числовой признак "возраст", сбрасываем дату рождения
   df['Age'] = Age_feature_creator(df)
   df.drop(['BirthDate'], axis='columns', inplace=True)
   # Вводим числовой признак "стаж", сбрасываем аналогичный категориальный
   df['NumValue'] = Num_value_feature_creator(df)
   df.drop(['JobStartDate', 'Value'], axis='columns', inplace=True)
   # Вводим числовой признак отношения платежа к доходу
   df['Payment_to_income'] = Payment_to_income(df)
   df.drop(['MonthProfit', 'MonthExpense', 'Loan_amount', 'Loan_term'],
           axis='columns', inplace=True)

   return df