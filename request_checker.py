import numpy as np
import pandas as pd


'''
Модуль, в котором реализуется преобразование полученного в
виде json фрейма с данными клиента к формату pd.DataFrame, для
последующей предобработки и обогащения новыми признаками.

Структура получаемых данных и кортежи возможных меток по
некоторым категориальным параметрам инициализированы в качестве
глобальных переменных для удобства внесения изменений при
поддержке.

Каждое из отдельных обрабатываемых полей (признаков) имеет свой
отдельный блок предобработчика try:...except:, которые необходимы
для обеспечения устойчивой работы приложения и исключения "падения"
приложения в случае проблем с данными. В блоках try: реализуются
попытки привести полученное значение к "нормальному" формату
(например, pd.DateTime для дат, float для чисел с плавающей запятой
и т.д.). В случае, если такое преобразование выполнить не удается,
соответствующему параметру присваивается значение np.NaN.

Данный модуль является полностью работоспособным, но не окончательным
вариантом, в дальнейшем будет редактироваться.
'''


REQUEST_STRUCTURE = (
    'BirthDate',
    'education',
    'employment status',
    'Value',
    'JobStartDate',
    'Position',
    'MonthProfit',
    'MonthExpense',
    'Gender',
    'Family status',
    'ChildCount',
    'SNILS',
    'Merch_code',
    'Loan_amount',
    'Loan_term',
    'Goods_category'
)
EDUCATION_VAL = (
    'Высшее - специалист',
    'Неоконченное среднее',
    'Среднее профессиональное',
    'Среднее',
    'Магистр',
    'Несколько высших',
    'Бакалавр',
    'Неоконченное высшее',
    'MBA',
    'Ученая степень'
)
EMPLOYMENT_STATUS_VAL = (
    'Работаю по найму полный рабочий день/служу',
    'Собственное дело',
    'Работаю по найму неполный рабочий день',
    'Студент',
    'Пенсионер',
    'Не работаю',
    'Декретный отпуск'
)
VALUE_VAL = (
    '0 месяцев 0 лет',
    '9 - 10 лет',
    '1 - 2 года',
    '10 и более лет',
    '2 - 3 года',
    '7 - 8 лет',
    '3 - 4 года',
    '5 - 6 лет',
    '4 - 5 лет',
    '6 - 7 лет',
    '6 месяцев - 1 год',
    '4 - 6 месяцев',
    '8 - 9 лет',
    'менее 4 месяцев'
)
FAMILY_STATUS_VAL = (
    'Никогда в браке не состоял(а)',
    'Женат / замужем',
    'Разведён / Разведена',
    'Гражданский брак / совместное проживание',
    'Вдовец / вдова'
)
GOODS_CATEGORY_VAL = (
    'Furniture',
    'Fitness',
    'Medical_services',
    'Education',
    'Other',
    'Travel',
    'Mobile_devices'
)


def check_n_fill(item) -> pd.DataFrame:
    '''Проверяем корректность данных, создаем из них pd.DataFrame'''
    # Получилось весьма громоздко, нужно будет ужать потом
    in_frame = pd.DataFrame(columns=REQUEST_STRUCTURE, index=[1])
    try:
        in_frame['BirthDate'] = pd.to_datetime(item.BirthDate)
    except Exception:
        in_frame['BirthDate'] = np.NaN
    try:
        if item.education in EDUCATION_VAL:
            in_frame['education'] = item.education
    except Exception:
        in_frame['education'] = np.NaN
    try:
        if item.employment_status in EMPLOYMENT_STATUS_VAL:
            in_frame['employment status'] = item.employment_status
    except Exception:
        in_frame['employment status'] = np.NaN
    try:
        if item.Value in VALUE_VAL:
            in_frame['Value'] = item.Value
    except Exception:
        in_frame['Value'] = np.NaN
    try:
        in_frame['JobStartDate'] = pd.to_datetime(item.JobStartDate)
    except Exception:
        in_frame['JobStartDate'] = np.NaN
    try:
        in_frame['Position'] = item.Position
    except Exception:
        in_frame['Position'] = np.NaN
    try:
        in_frame['MonthProfit'] = float(item.MonthProfit)
    except Exception:
        in_frame['MonthProfit'] = np.NaN
    try:
        in_frame['MonthExpense'] = float(item.MonthExpense)
    except Exception:
        in_frame['MonthExpense'] = np.NaN
    try:
        in_frame['Gender'] = int(item.Gender)
    except Exception:
        in_frame['Gender'] = np.NaN
    try:
        if item.Family_status in FAMILY_STATUS_VAL:
            in_frame['Family status'] = item.Family_status
    except Exception:
        in_frame['Family status'] = np.NaN
    try:
        in_frame['ChildCount'] = int(item.ChildCount)
    except Exception:
        in_frame['ChildCount'] = np.NaN
    try:
        in_frame['SNILS'] = int(item.SNILS)
    except Exception:
        in_frame['SNILS'] = np.NaN
    try:
        in_frame['Merch_code'] = int(item.Merch_code)
    except Exception:
        in_frame['Merch_code'] = np.NaN
    try:
        in_frame['Loan_amount'] = float(item.Loan_amount)
    except Exception:
        in_frame['Loan_amount'] = np.NaN
    try:
        in_frame['Loan_term'] = int(item.Loan_term)
    except Exception:
        in_frame['Loan_term'] = np.NaN
    try:
        if item.Goods_category in GOODS_CATEGORY_VAL:
            in_frame['Goods_category'] = item.Goods_category
    except Exception:
        in_frame['Goods_category'] = np.NaN

    return in_frame
