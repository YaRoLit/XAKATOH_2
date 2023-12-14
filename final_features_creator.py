import pandas as pd
import numpy as np
from preproc_position import position_preproc_by_Anna
import csv
import json
from dateutil.relativedelta import relativedelta
from datetime import datetime


def professional_risk_feature_creator(position) -> str:
    '''
    Calculates "Professional risk" column value based on risks.csv values.
    If position is not found sets risk by default to medium level=1.
    Risk levels:
    0 - Low
    1 - Medium
    2 - High
    :param position: input position
    :return: professional risk level
    '''
    risk = "1"
    position = position.lower().strip()

    with open('./data/risks.csv') as f:
        next(f)  # Skip the header
        reader = csv.reader(f, skipinitialspace=True, delimiter=';')
        risks = dict(reader)

    for pr in risks:
        pr = pr.lower().strip()
        if position == pr and (pr in risks):
            risk = risks[pr]

    return risk.strip()


def self_employed_feature_creator(position) -> int:
    '''
    Calculates "SelfEmployed" column value.
    Sets value to 1 if position contains "самоз" substring.
    :param position: input position
    :return: 0 if position doesn't contain "самоз" otherwise 1
    '''
    selfEmployed = 0
    position = position.lower().strip()

    if "самоз" in position:
        selfEmployed = 1

    return selfEmployed


def years_on_current_job_feature_creator(row) -> str:
    '''
    Calculates how long the applicant is working on the current job position
    as a difference in years of current date and "JobStartDate" value.
    :param: input dataframe row
    :return: group name according to years experience
    '''
    employment_status_ignored = ['Пенсионер', 'Студент',
                                 'Декретный отпуск', 'Не работаю']

    years_working_on_current_job = 0

    if not row["employment status"] in employment_status_ignored:
        end_date = datetime.now()
        start_date = row["JobStartDate"]
        years_working_on_current_job = relativedelta(
            end_date, start_date).years

    if years_working_on_current_job in range(-9999, 3):
        return '0-3 года'
    elif years_working_on_current_job in range(3, 6):
        return '3-6 лет'
    elif years_working_on_current_job in range(6, 10):
        return '6-10 лет'
    elif years_working_on_current_job in range(10, 9999):
        return 'больше 10 лет'


def age_group_feature_creator(row) -> str:
    '''
    Calculates applicant age group.
    :param: input dataframe row
    :return: age group name
    '''
    end_date = datetime.now()
    start_date = row["BirthDate"]
    age = relativedelta(end_date, start_date).years

    if age in range(-9999, 18):
        return '0-18'
    elif age in range(18, 21):
        return '18-21'
    elif age in range(21, 25):
        return '21-25'
    elif age in range(25, 45):
        return '25-45'
    elif age in range(45, 55):
        return '45-55'
    elif age in range(55, 75):
        return '55-75'
    elif age in range(75, 85):
        return '75-85'
    elif age in range(85, 9999):
        return '85+'


def emi_feature_creator(row) -> float:
    """
    Calculates Equated Monthly Installment(EMI).
    :param: input dataframe row
    :return: calculated EMI value
    """
    return round(row['Loan_amount']/row['Loan_term'], 2)


def balance_income_feature_creator(row) -> float:
    """
    Calculates the balance of the monthly salary after deducting expenses
    and the monthly loan payment (EMI).
    :param: input dataframe row
    :return: calculated income balance
    """
    return round(row['MonthProfit'] - row['MonthExpense'] - row['EMI'], 2)


# =============================================================================
# Пайплайн для сбора новых фич для используемой модели
# =============================================================================
def features_creator_pipe(df: pd.DataFrame) -> pd.DataFrame:
    '''Пайплайн  создания признаков, необходимых для работы модели'''
    df = df.copy()

    # Преобразование столбца Position
    df['Position'] = df.Position.apply(position_preproc_by_Anna)

    # Создание столбца с уровнем профессионального риска.
    df['ProfRisk'] = df.Position.apply(professional_risk_feature_creator)

    # Создаем признак самозанятости.
    df['SelfEmployed'] = df.Position.apply(self_employed_feature_creator)

    # Создаем признак стажа работы на текущем месте.
    df['YearsWorkedOnCurrentJob'] = df.apply(
        years_on_current_job_feature_creator, axis=1)

    # Создаем возрастные группы.
    df["AgeGroup"] = df.apply(age_group_feature_creator, axis=1)

    # Создаем ежемесячный платеж по кредиту.
    df['EMI'] = df.apply(emi_feature_creator, axis=1)

    # Создаем баланс доходов.
    df['BalanceIncome'] = df.apply(balance_income_feature_creator, axis=1)

    # Преобразуем код продавца в категорию.
    # df['Merch_code'] = df['Merch_code'].astype(np.int32)
    df['Merch_code'] = pd.Categorical(df['Merch_code'])

    # Сбрасываем все исходные признаки, коррелированные с новыми
    df = df.drop(['MonthProfit', 'MonthExpense', 'Loan_amount', 'Loan_term',
                 'JobStartDate', 'BirthDate'], axis=1)

    df['SNILS'] = df['SNILS'].astype(np.int16)
    df['Gender'] = df['Gender'].astype(np.int16)
    df['ChildCount'] = df['ChildCount'].astype(np.int16)
    df['ProfRisk'] = df['ProfRisk'].astype(np.int16)

    df = pd.get_dummies(df)
    df = fill_dummy_columns(df)

    return df


# В силу того, что при обучении нашей модели мы используем SMOTE
# балансировку данных, а она требует использование get_dummies(),
# модели были обучены с добавлением dummies столбцов. Данная функция
# восстанавливает  недостающие столбцы. Конечно, неправильный подход
# и по логике надо использовать OneHotEncoder, но это лишь временный
# костыль, который планируется исправить в дальнейшем.
def fill_dummy_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cat_feature_names = ['education', 'employment status', 'Value', 'Position',
                         'Family status', 'Merch_code', 'Goods_category',
                         'YearsWorkedOnCurrentJob', 'AgeGroup']

    with open('./data/dummies.json') as cat_dict_dump:
        cat_dict = json.load(cat_dict_dump)

    cols_to_add = []
    for col_name in cat_feature_names:
        matched_col = df.columns[df.columns.str.startswith(pat=col_name)]
        if (len(matched_col) > 0):
            matched_col_name = str(matched_col[0])
            if col_name in cat_dict:
                list_unique_cols = cat_dict[col_name]
                for uc in list_unique_cols:
                    un = f"{col_name}_{uc}"
                    if matched_col_name != un:
                        new_col_name = f"{col_name}_{uc}"
                        cols_to_add.append(new_col_name)
    df = df.reset_index()
    data_dict = dict.fromkeys(cols_to_add, [0])
    new_df = pd.DataFrame(data=data_dict, columns=cols_to_add)
    result = pd.concat([df, new_df], axis=1)
    return result
