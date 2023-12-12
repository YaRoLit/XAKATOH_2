import os
import pandas as pd
import pickle
from catboost import CatBoostClassifier
from catboost import CatBoostRegressor

from preproc_position import position_preproc_by_Igor


'''
В данном модуле реализован конвейер заполнения пропусков
в полученном в post запросе фрейме данных с параметрами клиента.

Конвейер реализует 4-х ступенчатую схему заполнения пропусков
для разных моделей пропусков по следующему алгоритму:
1. Заполняются "не случайные" пропуски в признаках 'Value' и
'Position' для безработных на специально определенные метки:
"0 месяцев 0 лет" и "Безработный" соответственно.
2. Заполняются пропуски в параметрах, имеющих формат pd.Datetime
на метку pd.Timestamp.max. В ходе предварительного исследования
было установлено, что это лучший вариант для работы модели.
3. Заполняются пропуски в данных на основе предобученных моделей.
Модели, которые используются для заполнения, были обучены на
исходном датасете. Они учитывают различные взаимосвязи во входных
данных и при исследовании показали себя точнее, чем заполнение
пропусков "простыми" стратегиями на основе статистик конкретного
признака датасета (не для всех параметров). Подробнее о работе
данного алгоритма описано в соответствующей функции ниже.
4. Все оставшиеся пропуски заполняются "простыми" стратегиями
с помощью обученных на исходном датасете SKLearn Simple Imputers.
'''


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
# Так как загружаемые экземпляры SimpleImputers используются
# при работе нескольких функций, они определяются как глобальные
# переменные и хранятся в памяти постоянно в ходе работы модуля.
cat_imputer = pickle.load(open('./models/simple.cat', 'rb'))
reg_imputer = pickle.load(open('./models/simple.reg', 'rb'))


def unemployed_nansfiller(df: pd.DataFrame) -> pd.DataFrame:
    '''Заполняем пропуски в JobStartDate, Value и Position для безработных'''
    # Список "безработных" определен как константа в глобальных
    # переменных для удобства внесения изменений в случае изменения списка.
    # Используются множественные условия, так как некоторые лица, являющиеся
    # студентами, имеют работу.
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
    # Перебираем датафрейм по столбцам
    for col in df.columns:
        # Создаем выборку, для которой в соответствующем столбце есть пропуски
        nans_frame = df[(df[col].isna())]
        # Если созданная выборка не пустая, то заполняем пропуски моделями
        if nans_frame.shape[0]:
            # Сначала делаем несколько преобразований только на время чистки
            # пустышек, преобразования нужны исключительно для работы моделей.
            # В частности, преобразуем столбец Position, так как на его метках
            # были обучены модели.
            nans_frame['Position'] = position_preproc_by_Igor(nans_frame)
            # Если в предикторах есть пропуски, заполняем их SimpleImputers
            nans_frame[CAT_COLUMNS] = cat_imputer.transform(
                                                nans_frame[CAT_COLUMNS])
            nans_frame[REG_COLUMNS] = reg_imputer.transform(
                                                nans_frame[REG_COLUMNS])
            # Переводим даты в формат "год", такой формат использовался при
            # обучении моделей.
            nans_frame['BirthDate'] = nans_frame['BirthDate'].dt.year
            nans_frame['JobStartDate'] = nans_frame['JobStartDate'].dt.year
            # Если в папке models есть модель с именем выбранного столбца,
            # то загружаем её. Для классификационных и регрессионных моделей
            # предусмотрены разные загрузчики.
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
            # Вот здесь уже вносим изменения в основной датафрейм
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
