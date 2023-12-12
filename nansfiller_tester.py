import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import warnings
import nans_filler

warnings.simplefilter(action='ignore')


'''
Простой калькулятор качества заполнения пропусков в исходном
датасете с помощью разных стратегий заполнения. Использует
./data/model_val.csv, который произвольно зашумляет np.NaN, а
потом "чистит" зашумленный датасет разными способами и выводит
метрики работы конвейеров очистки. В качестве 0-й базовой
стратегии выводит метрики заполнения SKLearn Simple Imputers
на основе статистических параметров датасета, в качестве
альтернативы подключен действующий пайплайн заполнения пропусков.
'''


CLS_FEATURES = ['education',
                'employment status',
                'Value',
                'Position',
                'Gender',
                'Family status',
                'ChildCount',
                'SNILS',
                'Merch_code',
                'Loan_term',
                'Goods_category']

REG_FEATURES = ['MonthProfit',
                'MonthExpense',
                'Loan_amount']


def cat_num_split(df: pd.DataFrame) -> tuple:
    '''Ищем категориальные и числовые признаки в датафрейме'''
    cat_columns = []
    num_columns = []
    for column_name in df.columns:
        if (df[column_name].dtypes == object):
            cat_columns += [column_name]
        elif column_name not in datetime_columns:
            num_columns += [column_name]

    return cat_columns, num_columns


def datetime_to_timestamp(date_time):
    '''Переводим дату-время в timestamp'''

    return date_time // 10 ** 9


def target_columns_dropper(df: pd.DataFrame) -> pd.DataFrame:
    '''Сбрасываем столбцы с целевыми переменными ответа банков'''
    df = df.copy()
    banks = ['BankA_decision',
             'BankB_decision',
             'BankC_decision',
             'BankD_decision',
             'BankE_decision']
    df.drop(banks, axis='columns', inplace=True)

    return df


def create_frame_with_nans(df: pd.DataFrame) -> tuple:
    '''Создаем зашумленный np.NaN датасет из исходного'''
    df = df.copy()
    nans_rows_index = {}
    for col in df.columns:
        sample_index = df.sample(round(df.shape[0] * 0.4)).index
        nans_rows_index[col] = sample_index
        df.loc[sample_index, col] = np.NaN

    return df, nans_rows_index


def simple_strategy_nansfiller(df: pd.DataFrame) -> pd.DataFrame:
    '''Заполняем np.NaN простыми стратегиями для примера'''
    df = df.copy()
    df[datetime_columns].fillna(df[datetime_columns].median(), inplace=True)
    cat_imputer = SimpleImputer(strategy='most_frequent')
    num_imputer = SimpleImputer(strategy='mean')
    df[CLS_FEATURES] = cat_imputer.fit_transform(df[CLS_FEATURES])
    df[REG_FEATURES] = num_imputer.fit_transform(df[REG_FEATURES])

    return df


def calc_nansfiller_quality_score(
        df_real: pd.DataFrame,
        df_pred: pd.DataFrame,
        nans_rows_index: dict) -> tuple:
    '''
    Рассчитываем оценку качества работы заполнителя пропусков.
    На вход подаем исходный датасет, не зашумленный пропусками
    + датасет, который был подвергнут обработке с заполненными
    NaNами, а также словарь с индексами пропусков.
    Метрика оценки рукописная и работает следующим образом:
    - для категориальных столбцов вычисляется количество строк,
    где истинное_значение == предсказанному_значению, после чего
    полученное число делится на общее количество строк в массиве.
    Либо можно использовать метрику Accuracy.
    - для числовых столбцов я остановился на среднем RMSE для всех
    числовых столбцов КМК это не вполне корректное решение.
    '''
    # Переводим столбцы datetime.dtypes в формат timestamp
    # Чтобы точнее считать метрику ошибки
    score = {}
    for col in CLS_FEATURES:
        y_true = df_real.loc[nans_rows_index[col], col]
        y_pred = df_pred.loc[nans_rows_index[col], col]
        score[col] = (y_true == y_pred)
        score[col] = score[col].sum()
        score[col] = str(f'{round(score[col], 4)} True')

    for col in REG_FEATURES:
        y_true = df_real.loc[nans_rows_index[col], col]
        y_pred = df_pred.loc[nans_rows_index[col], col]
        score[col] = mean_squared_error(y_true, y_pred) ** 0.5
        score[col] = str(f'{round(score[col])} RMSE')

    return score


def result_shower(*score_dicts) -> None:
    '''Выводим сравнительную таблицу по испытаниям пайплайнов'''
    print('==================================================================')
    print(f'В df заполнено NaN {round(df.shape[0] * 0.4)} значений в столбцах')
    print('==================================================================')
    score_table = pd.DataFrame(index=score_dicts[0].keys())
    for idx, score in enumerate(score_dicts):
        score_table[idx] = score.values()
    print(score_table)


if __name__ == "__main__":
    datetime_columns = ['BirthDate', 'JobStartDate']
    df = pd.read_csv(
        filepath_or_buffer='./data/model_val.csv',
        sep=';',
        index_col='SkillFactory_Id',
        parse_dates=datetime_columns
        )
    # Сбрасываем столбцы с целевыми переменными
    df_real = target_columns_dropper(df)
    # Создаем выборку, зашумленную NaNs, и их индексы по колонкам
    df_test, nans_rows_index = create_frame_with_nans(df_real)
    # Создаем первую очищенную выборку при помощи простых стратегий заполнения
    df_simple = simple_strategy_nansfiller(df_test)
    # Считаем метрику качества заполнения простой стратегией
    score_simple_strategy = calc_nansfiller_quality_score(
        df_real, df_simple, nans_rows_index)
    # Создаем очищенную выборку при помощи Yaro стратегии заполнения
    df_Yaro = nans_filler.fill_nans_pipe(df_test)
    # Считаем метрику качества заполнения Yaro стратегией
    score_Yaro_strategy = calc_nansfiller_quality_score(
        df_real, df_Yaro, nans_rows_index)

    result_shower(score_simple_strategy, score_Yaro_strategy)
