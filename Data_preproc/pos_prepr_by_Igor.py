"""
 Файл содержит функции для очистки данных первых 6 столбцов датасета и создания новых признаков.

 Чтобы импортировать эти функции в свой файл используйте строку:
 from pos_prepr_by_Igor import *

 Если файл лежит в отдельной папке, то нужно сделать примерно так (подставив нужный путь):

 import sys 
 import os
 sys.path.append(os.path.abspath("/home/igor/Plastov/XAKATOH_2/Data_preproc"))
 from pos_prepr_by_Igor import *
"""


import pandas as pd
import numpy as np



def drop_completely_nan_rows(df:pd.DataFrame) -> pd.DataFrame:
    """
    Функция убирает строки таблицы целиком состоящие из nan.
    """

    d = df.copy()

    # Убираем строки таблицы целиком состоящие из nan.
    d = d.dropna(axis=0, how='all')
   
    return d

# ### Очистка признака *BirthDate*
#BirthDate

def BirthDate_feature_cleaner(df:pd.DataFrame)->pd.Series:
    """
    Проверка и заполнение модой пропусков фичи 'BirthDate'.
    """
    mc = df.copy()
    mc = drop_completely_nan_rows(mc)
    fch = 'BirthDate' # Имя фичи.
    f = pd.to_datetime(mc[fch], format='%Y-%m-%d')
    bad_rows = len(f[f.isna()])
    m = f.mode()
    print("Mode =", m)
    f[f.isna()] = m 
    
    print(f"Количество nan в фиче <{fch}>: {bad_rows} штук.")
    if bad_rows == 0:
        print(f'Фича <{fch}> не содержит пропусков.')
    return f

def test_BirthDate_feature_cleaner(df):
    md = df.copy()
    md = drop_completely_nan_rows(md)
    md.loc[10,'BirthDate'] = np.nan # Портим одно значение затем проверяем, что оно было исправлено.
    print(md.iloc[10]['BirthDate'] )
    res = BirthDate_feature_cleaner(md)
    print(res)
    print(type(res))

#test_BirthDate_feature_cleaner(df)

def get_zodiac(month: int, date:int)->str:
   value="просто животное"
   if   ((month==1)  and (date>=20)) or ((month==2 ) and (date<=18)): value="Водолей"
   elif ((month==2)  and (date>=19)) or ((month==3 ) and (date<=20)): value="Рыбы"
   elif ((month==3)  and (date>=21)) or ((month==4 ) and (date<=19)): value="Овен"
   elif ((month==4)  and (date>=20)) or ((month==5 ) and (date<=20)): value="Телец"
   elif ((month==5)  and (date>=21)) or ((month==6 ) and (date<=21)): value="Близнецы"
   elif ((month==6)  and (date>=22)) or ((month==7 ) and (date<=22)): value="Рак"
   elif ((month==7)  and (date>=23)) or ((month==8 ) and (date<=22)): value="Лев"
   elif ((month==8)  and (date>=23)) or ((month==9 ) and (date<=22)): value="Дева"
   elif ((month==9)  and (date>=23)) or ((month==10) and (date<=22)): value="Весы"
   elif ((month==10) and (date>=23)) or ((month==11) and (date<=21)): value="Скорпион"
   elif ((month==11) and (date>=22)) or ((month==12) and (date<=21)): value="Стрелец"
   elif ((month==12) and (date>=22)) or ((month==1 ) and (date<=19)): value="Козерог"
   return value

def test_zodiac():
    print(zodiac(1,21))
    print(zodiac(11,11))
    print(zodiac(1,19))

#test_zodiac()

def zodiac_feature_creator(df:pd.DataFrame)->pd.Series:
    """
    Фича содержит знаки зодиака соответствующие дню и месяцу рождения соискателя кредита.
    """

    mc = df.copy()
    f = BirthDate_feature_cleaner(mc)
    fch = 'BirthDate' # Имя фичи.
    
    bad_rows = len(f[f.isna() == True])

    zodiac = f.apply(lambda x: get_zodiac(x.month, x.day))

    print(f"Количество nan в фиче <{fch}>: {bad_rows} штук.")
    if bad_rows == 0:
        print(f'Фича <{fch}> не содержит пропусков.')
    return zodiac

def test_zodiac_feature_creator():
    md = df.copy()
    md = drop_completely_nan_rows(md)
    res = zodiac_feature_creator(md)
    print(res)

#test_zodiac_feature_creator()

# ### Очистка признака *education*

#education

def education_feature_cleaner(df:pd.DataFrame)->pd.Series:
    """
    Проверка и очистка фичи 'education'.
    Преобразуем образование в числовые шкалу от 0.1 до 1.,
    чем больше тем лучше образование.
    """
    mc = df.copy()

    fch = 'education' # Имя фичи.
    f = mc[fch]
    f = f.str.lower()
    f = f.str.strip()
    bad_rows = len(f[f.isna()])

    mode = f[~f.isna()].mode() 
    f[f.isna()] = mode


    print(f"Количество nan в фиче <{fch}>: {bad_rows} штук.")
    if bad_rows == 0:
        print(f'Фича <{fch}> не содержит пропусков.')
    return f

def test_education_feature_cleaner():
    md = df.copy()
    md = drop_completely_nan_rows(md)
    md.loc[10,'education'] = np.nan # Портим одно значение затем проверяем, что оно было исправлено.
    print(md.iloc[10]['education'] )
    res = education_feature_cleaner(md)
    print(res)

#test_education_feature_cleaner()


def educationDig_feature_creator(df:pd.DataFrame)->pd.Series:
    """
    Создание новой шкальной фичи из 'educationDig' фичи 'education'.
    Преобразуем образование в числовые шкалу от 0.1 до 1.,
    чем больше тем лучше образование.
    """
    mc = df.copy()
    f = education_feature_cleaner(mc)

    fch = 'education' # Имя фичи.
    bad_rows = len(f[f.isna() == True])

    # Преобразуем образование в числовые шкалу, чем больше тем лучше.
    rate_dict={
     'высшее - специалист': 0.6,
     'среднее профессиональное': 0.3,
     'среднее' :0.2,
     'неоконченное высшее' : 0.4,
     'бакалавр' : 0.5,
     'несколько высших' : 0.8,
     'магистр' : 0.7,
     'неоконченное среднее': 0.1,
     'mba'  : 0.7,
     'ученая степень': 1.
     }

    # Оцифровываем уровень образования.
    f = f.replace(rate_dict, regex=True)

    print("Легенда по которой выполняется оцифровка уровня образования:\n",
          rate_dict)

    print(f"Количество nan в фиче <{fch}>: {bad_rows} штук.")
    if bad_rows == 0:
        print(f'Фича <{fch}> не содержит пропусков.')
    return f

def test_educationDig_feature_creator():
    md = df.copy()
    md = md.drop_compleately_nan_rows(md)
    
    res = educationDig_feature_creator(md)
    print(res)

#test_educationDig_feature_creator()

# ### Очистка признака *employment status*

#employment_status

def employment_status_feature_cleaner(df:pd.DataFrame)->pd.Series:
    """
    Проверка и очистка фичи 'employment status'.
    Пропуски заполняются модой.
    """

    mc = df.copy()
    fch = 'employment status' # Имя фичи.
    f = df[fch].copy()
    initial_bad_rows = len(f[f.isna()])
    f = f.str.lower()
    f = f.str.strip()
    bad_rows = len(f[f.isna()])
    
    mode = f[~f.isna()].mode()
    f[f.isna()] = mode

    print(f"Исходное количество nan в фиче <{fch}>: {initial_bad_rows} штук.")
    print(f"Устранено: {initial_bad_rows - bad_rows} штук.")
    if bad_rows == 0:
        print(f'Фича <{fch}> не содержит пропусков.')
    return f

def test_employment_status_feature_cleaner():
    md = df.copy()
    md = md.drop_compleately_nan_rows(md)
    md.loc[10,'employment status'] = np.nan # Портим одно значение затем проверяем, что оно было исправлено.
    print(md.iloc[10]['employment status'] )
    res = employment_status_feature_cleaner(md)
    print(res)

#test_employment_status_feature_cleaner()


def employment_statusDig_feature_creator(df:pd.DataFrame)->pd.Series:
    """
    Преобразует 'employment status' (занятость) в числовую шкалу от 0.001 до 1.,
    чем больше, тем занятость полнее.
    """

    mc = df.copy()
    f = employment_status_feature_cleaner(mc)
    fch = 'employment status' # Имя фичи.

    # Преобразуем занятость в числовую шкалу от 0.001 до 1.,
    # чем больше, тем занятость полнее.
    rate_dict={
    'работаю по найму полный рабочий день/служу': 1.,
    'собственное дело': 1.,
    'не работаю': 0.001,
    'работаю по найму неполный рабочий день': 0.5,
    'студент': 0.001,
    'декретный отпуск': 0.001,
    'пенсионер':0.001
    }

    # Оцифровываем уровень образования.
    f = f.replace(rate_dict, regex=True)

    print("Легенда по которой выполняется оцифровка уровня занятости:\n",
          rate_dict)

    return f


def test_employment_statusDig_feature_creator():
    md = df.copy()
    md = md.drop_compleately_nan_rows(md)
    md.loc[10,'employment status'] = np.nan # Портим одно значение затем проверяем, что оно было исправлено.
    print(md.iloc[10]['employment status'] )
    res = employment_statusDig_feature_creator(md)
    print(res[res < 0.6])

#test_employment_statusDig_feature_creator()    

# ### Очистка признака  *Value*

def Value_feature_cleaner(df:pd.DataFrame)->pd.Series:
    """
    Проверка и очистка фичи 'Value'.
    пропуски трудового стажа  заполняются медианным значением.
    """
    mc = df.copy()

    fch = 'Value' # Имя фичи.
    f = mc[fch]
    f = f.str.lower()
    f = f.str.strip()
    
    mask = f.isna()
    initial_bad_rows = len(f[mask])

    mode = f[~mask].mode()
    f[mask] = mode

    bad_rows = len(f[f.isna() == True])
    print(f"Исходное количество nan в фиче <{fch}>: {initial_bad_rows} штук.")
    print(f"Устранено: {initial_bad_rows - bad_rows} штук.")
    if bad_rows == 0:
        print(f'Фича <{fch}> не содержит пропусков.')
    return f

def test_Value_feature_cleaner():
  md = df
  md = md.drop_compleately_nan_rows(md)
  md.loc[10,'Value'] = np.nan # Портим одно значение затем проверяем, что оно было исправлено.
  print(md.iloc[10]['Value'] )
  res = Value_feature_cleaner(md)
  print(res.value_counts())

#test_Value_feature_cleaner()

def ValueDig_feature_creator(df:pd.DataFrame)->pd.Series:
    """
    Проверка и очистка фичи 'Value'.
    Преобразует трудовой стаж в числовую шкалу от 0.001 до 1.,
    чем больше, тем занятость полнее.
    NaN - заполняются средним значением по соответствующей возрастной группе.
    для студентов NaN заменяется минимальным стажем.
    """
    mc = df.copy()

    f = Value_feature_cleaner(mc)
    fch = 'Value' # Имя фичи.
    
    d = pd.to_datetime(mc['BirthDate'])
    mc['Year'] = d.dt.year

    mc['Year'] = mc['Year'].astype(int)

    # Преобразуем трудовой стаж в числовую шкалу от 0.017 до 1.,
    # чем больше, тем занятость полнее.
    rate_dict={
    '10 и более лет': 1.0,
    '3 - 4 года':  0.35,
    '2 - 3 года':  0.25,
    '4 - 5 лет':   0.45,
    '5 - 6 лет':   0.55,
    '1 - 2 года':  0.15,
    '6 - 7 лет':   0.65,
    '7 - 8 лет':   0.75,
    '8 - 9 лет':   0.85,
    '6 месяцев - 1 год': 0.075,
    '9 - 10 лет': 0.95,
    '4 - 6 месяцев': 0.042,
    'менее 4 месяцев': 0.017
    }

    # Оцифровываем уровень образования.
    f = f.replace(rate_dict, regex=True)

    print("Легенда по которой выполняется оцифровка трудового стажа:\n",
          rate_dict)

    mc[fch] = f
    
    years = mc['Year'].to_list()
    for y in years:
      m = mc[mc['Year'] == y][fch].mean()
      mask = (mc[fch].isna()) & ( mc['Year'] == y)
      f[mask] = m

    return f

def test_ValueDig_feature_creator():
  md = df.copy()
  md = md.drop_compleately_nan_rows(md)
  md ['BirthDate'] = BirthDate_feature_cleaner(md)
  res = ValueDig_feature_creator(md)
  print(res.value_counts())

#test_ValueDig_feature_creator()

# ### Очистка признака *JobStartDate*

#JobStartDate

def JobStartDate_feature_cleaner(df:pd.DataFrame)->pd.Series:
    """
    Проверка и очистка фичи 'JobStartDate'.
    Вместо NaN подставляется дата рождения увеличенная на случайное целое 17..23 лет.
    """
    mc = df.copy()

    fch = 'JobStartDate' # Имя фичи.

    f = mc[fch]
    mc['syntJobStartDate'] = pd.to_datetime(mc['BirthDate'])
    mc['syntJobStartDate'] = pd.to_datetime(mc['syntJobStartDate'], format='%Y-%m-%d')
    mc['syntJobStartDate'] =  mc['syntJobStartDate'] + pd.offsets.DateOffset(years=(17 + np.random.randint(6)))
    mask = f.isna()

    
    initial_bad_rows = len(f[mask])
    f[mask] = mc['syntJobStartDate']
    f = pd.to_datetime(f)
    f = pd.to_datetime(f, format='%Y-%m-%d')
    bad_rows = len(f[f.isna()])
    print(type(f)) 
    print(f"Исходное количество nan в фиче <{fch}>: {initial_bad_rows} штук.")
    print(f"Устранено: {initial_bad_rows - bad_rows} штук.")
    if bad_rows == 0:
        print(f'Фича <{fch}> не содержит пропусков.')
    return f

def test_JobStartDate_feature_cleaner():
   md =df.copy()
   md = md.drop_compleately_nan_rows(md)
   f = JobStartDate_feature_cleaner(md)
   print("res=", f)

#test_JobStartDate_feature_cleaner()

def CarierLevel_feature_creator(df:pd.DataFrame)->pd.Series:
    """
    Категорийный признак "Карьерный уровень клиента".
    Возвращает примерно такой состав данных:
    НЕТ_ДАННЫХ       2293
    НИЗШЕЕ_ЗВЕНО     1385
    СРЕДНЕЕ_ЗВЕНО    1104
    МЕН1              907
    НЕРАЗОБРАНО       752
    ДИР2              533
    МЕН2              523
    САМ               520
    ИП                390
    ДИР1              226
    ЗАМДИР            124
    ЗАМНАЧ             30
    Name: Position, dtype: int64
    """

    f = df['Position'].copy()
    f = f.str.lower()
    f = f.str.strip()
    f1 = f

    L_7 = [
    "генеральный директор",
    "генеральный дирекор",
    "генеральный",
    "гене",
    "генер",
    "генера",
    "ген дир",
    "ген",
    "учредитель",
    'полномочный представитель президента'
    ]

    for s in L_7:
       f1 = f1.mask(lambda  x : x == s, other = 'ДИР1')

    L_6 = [
    "директор",
    "руководитель",
    "директор филиала/ департамента",
    'директор магазина',
    'дир',
    'дире'
    ]

    for s in L_6:
       f1 = f1.mask(lambda  x : x == s, other = 'ДИР2')

    L_5 = [
    "заместитель директора",
    "заместитель генерального директора",
    'заместитель директора',
    "зам",
    "коммерческий директор",
    "заместитель / и.о. генерального директора",
    "и.о. заместитель ген. директора по развитию",
    "зам.руководителя"
    ]

    for s in L_5:
       f1 = f1.mask(lambda  x : x == s, other = 'ЗАМДИР')


    L_4 = [
      "старший менеджер",
      "старший мене",
      "старший менедж",
      "ведущий менеджер",
      "главный менеджер",
      "главный специалист",
      'главный спе'
      "главный инженер",
      "менеджер",
      "мене",
      "менед",
      "руков",
      "руко",
      "рук",
      "главный бухгалтер",
      'главный бух',
      "начальник отдела",
      "начальник",
      "продюсер",
      "руководитель отдела",
      "менеджер по работе с ключевыми клиентами",
      "начальнык отдела",
      "начальник п",
      'менеджер по работе с маркетплейсами',
      'руководитель колл-центра',
      'начальник участка',
      'руководитель отдела продаж',
      'региональный менеджер',
      'директор по развитию',
      'заведующая отделением',
      'старший администратор'
    ]

    for s in L_4:
       f1 = f1.mask(lambda  x : x == s, other = 'МЕН1')

    L_3 = [
     "администратор",
     "админи",
     "адм",
     "вдминистратор",
     "управляющий",
     "управляющая",
     "менеджер по продажам",
     "специалист по продажам",
     "менедж",
     "мастер",
     "маст",
     "оператор склада",
     "менеджер по работе с маркетплейсами",
     'руководитель колл-центра', 'заведующий складом',
     'менеджер по закупкам','менеджер по закупке',
     'старший продавец',
     'менеджер торговой зоны',
     'старший сотрудник охраны',
     'командир отдедения',
     'командир отделения',
     'старший кладовщик',
     'старший продавец-консультант',
     'заведующий'
     ]

    for s in L_3:
       f1 = f1.mask(lambda  x : x == s, other = 'МЕН2')


    L_2 = [
   "старший специалист",
   "старший спец",
   "ведущий специалист",
   'ведущий спе',
   'логопед',
   'психолог',
   'воспит',
   'дизайнер',
   "товаровед",
   "маркетолог",
   "бухгалтер",
   'бухга',
   'бухг'
   "бух",
   'бух',
   "преподаватель",
   "препо",
   "учитель",
   "инженер",
   "инж",
   "оператор",
   "врач",
   "программист",
   "графический дизайнер",
   "репетитор",
   "технолог",
   "экономист",
   "юрист",
   "адвокат",
   "председатель",
   "машинист",
   "хормейстер",
   "ведущий инженер",
   "старший слесарь",
   "системный администратор",
   "воспитатель",
   "эксперт",
   "доцент",
   "финансовый консультант",
   "аналитик",
   'средний медперсонал',
   'маркето',
   'режиссер',
   'режиссер',
   'музыкальный р',
   'кассир-контролер',
   'секретарь',
   'риэлтор',
   'педагог',
   'пластический хирург'

   ]
    for s in L_2:
      f1 = f1.mask(lambda  x : x == s, other = 'СРЕДНЕЕ_ЗВЕНО')

    L_1 = [
     "специалист",
     "спец",
     "специ",
     "водитель",
     "продавец",
     "прода",
     "повар",
     "официант",
     "курьер",
     "консультант",
     "продавец-консультант",
     "продавец консультан",
     "продавец-кассир",
     "кладовщик",
     "косметолог",
     "сотрудник охраны",
     "слесарь",
     "сле",
     "кассир",
     "массажист",
     "электромонтер",
     "монтажник",
     "мастер маникюра",
     "электрик",
     "элек",
     "торговый представитель",
     "механик",
     "супервайзер",
     "инструктор",
     "персональный менеджер",
     "водитель-экспедитор",
     "водит",
     "продавец кассир",
     "рабочий",
     'сварщик',
     'свар',
     'агент',
     'диспетчер',
     'медицинская сестра',
     'медсестра',
     'токарь',
     "офи",
     'тренер',
     'владелец',
     'психолог-консультант',
     'помощник юриста',
     "швея",
     "парикмахер",
     "техник",
     'охранник',
     'няня',
     'косметолог-визажист',
     'сотрудник',
     'сотрудник склада',
     'кондитер',
     'фельдшер',
     'формовщик',
     'маляр',
     'проходчик',
     'техперсонал',
     'строитель',
     'клад',
     'наставник',
     'уборщица',
     'оптометрист',
     'пекарь',
     'официа',
     'контралер',
     'разнорабочая',
     'кладов',
     'комплектовщик',
     'термист'
     ]

    for s in L_1:
      f1 = f1.mask(lambda  x : x == s, other = 'НИЗШЕЕ_ЗВЕНО')

    f1 = f1.mask(lambda x : x == "индивидуальный предприниматель", other = 'ИП')
    f1 = f1.mask(lambda x : x == "инд", other = 'ИП')
    f1 = f1.mask(lambda x : x == "ип", other = 'ИП')

    f1 = f1.mask(lambda x : x == "самозанятый", other = 'САМ')
    f1 = f1.mask(lambda x : x == "самозанятая", other = 'САМ')


    f1=f1.fillna('НЕТ_ДАННЫХ')

    Filled = ['ДИР1','ДИР2', 'МЕН1', 'МЕН2','ИП','САМ',
              'ЗАМДИР','ЗАМНАЧ', 'НИЗШЕЕ_ЗВЕНО', 'СРЕДНЕЕ_ЗВЕНО', 'НЕТ_ДАННЫХ']

    # Названия должностей которые входят в некоторую строку.
    f1[f1.str.contains('рабочий')] = 'НИЗШЕЕ_ЗВЕНО'
    f1[f1.str.contains('слесарь')] = 'НИЗШЕЕ_ЗВЕНО'
    f1[f1.str.contains('помощник')] = 'НИЗШЕЕ_ЗВЕНО'
    f1[f1.str.contains('сборщик')] = 'НИЗШЕЕ_ЗВЕНО'
    f1[f1.str.contains('руководитель')] = 'МЕН1'
    f1[f1.str.contains('начальник ')] = 'МЕН1'
    f1[f1.str.contains('самоз')] = 'САМ'
    f1[f1.str.contains('инспектор')] = 'СРЕДНЕЕ_ЗВЕНО'
    f1[f1.str.contains('инженер')] = 'СРЕДНЕЕ_ЗВЕНО'
    f1[f1.str.contains('врач')] = 'СРЕДНЕЕ_ЗВЕНО'
    f1[f1.str.contains('вра')] = 'СРЕДНЕЕ_ЗВЕНО'
    f1[f1.str.contains('фармацевт')] = 'СРЕДНЕЕ_ЗВЕНО'
    f1[f1.str.contains('старший специалист')] = 'СРЕДНЕЕ_ЗВЕНО'
    f1[f1.str.contains('представитель')] = 'СРЕДНЕЕ_ЗВЕНО'
    f1[f1.str.contains('преподаватель')] = 'СРЕДНЕЕ_ЗВЕНО'
    f1[f1.str.contains('ведущий')] = 'СРЕДНЕЕ_ЗВЕНО'
    f1[f1.str.contains('бухг')] = 'СРЕДНЕЕ_ЗВЕНО'
    f1[f1.str.contains('учитель')] = 'СРЕДНЕЕ_ЗВЕНО'
    f1[f1.str.contains('заместитель директора')] = 'ЗАМДИР'
    f1[f1.str.contains('заместитель')] = 'ЗАМНАЧ'
    f1[f1.str.contains('заме')] = 'ЗАМНАЧ'
    f1[f1.str.contains('оператор')] = 'НИЗШЕЕ_ЗВЕНО'
    f1[f1.str.contains('лаборант')] = 'НИЗШЕЕ_ЗВЕНО'
    f1[f1.str.contains('электрик')] = 'НИЗШЕЕ_ЗВЕНО'
    f1[f1.str.contains('продавец')] = 'НИЗШЕЕ_ЗВЕНО'
    f1[f1.str.contains('сварщик')] = 'НИЗШЕЕ_ЗВЕНО'
    f1[f1.str.contains('менеджер')] = 'МЕН2'
    f1[f1.str.contains('главный специалист')] = 'МЕН2'
    f1[f1.str.contains('специалист')] = 'НИЗШЕЕ_ЗВЕНО'
    f1[f1.str.contains('монтажник')] = 'НИЗШЕЕ_ЗВЕНО'
    f1[f1.str.contains('водитель ')] = 'НИЗШЕЕ_ЗВЕНО'
    f1[f1.str.contains('парикмахер')] = 'НИЗШЕЕ_ЗВЕНО'
    f1[f1.str.contains('ногтевого')] = 'НИЗШЕЕ_ЗВЕНО'
    f1[f1.str.contains('санитарка')] = 'НИЗШЕЕ_ЗВЕНО'
    f1[f1.str.contains('продав')] = 'НИЗШЕЕ_ЗВЕНО'
    f1[f1.str.contains('монтер')] = 'НИЗШЕЕ_ЗВЕНО'
    f1[f1.str.contains('техник')] = 'НИЗШЕЕ_ЗВЕНО'
    f1[f1.str.contains('директор')] = 'ДИР2'


    #f1 = f1[~f1.isin(Filled)] # Временно убираем обработанные строки.
    f1[~f1.isin(Filled)] = "НЕРАЗОБРАНО"

    #print(contain_values)
    CarierLevel = f1
    return CarierLevel

def test_CarierLevel_feature_creator():
   md =df.copy()
   md = md.drop_compleately_nan_rows(md)
   f = CarierLevel_feature_creator(md)
   print("res=", f.value_counts())


#test_CarierLevel_feature_creator()

def CarierLevelDig_feature_creator(df:pd.DataFrame)->pd.Series:
    """
     Числовой признак "Карьерный уровень клиента", чем выше тем лучше.
     Состав выходной колонки данных примерно такой:
     1    2293
     2    2137
     3    2014
     5     907
     7     533
     4     523
     9     226
     8     124
     6      30
    """

    f1 = CarierLevel_feature_creator(df)
    rate_dict={
     'НЕРАЗОБРАНО': 2,
     'НЕТ_ДАННЫХ': 1,
     'НИЗШЕЕ_ЗВЕНО': 2,
     'САМ': 3,
     'ИП': 3,
     'СРЕДНЕЕ_ЗВЕНО': 3,
     'МЕН2': 4,
     'МЕН1': 5,
     'ЗАМНАЧ': 6,
     'ДИР2': 7,
     'ЗАМДИР': 8,
     'ДИР1': 9}

    f1 = f1.replace(rate_dict, regex=True)
    return f1

def test_CarierLevelDig_feature_creator():
   md =df.copy()
   md = md.drop_compleately_nan_rows(md)
   f = CarierLevelDig_feature_creator(md)
   print("res=", f.value_counts())

#test_CarierLevelDig_feature_creator()

def CarierVelocity_feature_creator(df:pd.DataFrame)->pd.Series:
   """
   Признак "Скорость карьерного роста" клиента (Отношение карьерного уровня
   к длительности карьеры).

   Состав выходной колонки данных примерно такой:
   0       0.487707
   1       7.019231
   2       0.388711
   3       0.388711
   4       0.388711
          ...
   8782         NaN
   8783         NaN
   8784         NaN
   8785         NaN
   8786         NaN
   """

   mc = df.copy()
   rates = CarierLevelDig_feature_creator(mc)
   values = ValueDig_feature_creator(mc)
   carier_velocity = rates/values
   return carier_velocity

def test_CarierVelocity_feature_creator():
  md = df.copy()
  md = drop_completely_nan_rows(md)
  print(CarierVelocity_feature_creator(md))

#test_CarierVelocity_feature_creator()


def clean_features(df:pd.DataFrame)->pd.DataFrame:
   """
   Функция выполняет очистку фич.
   """

   df = drop_completely_nan_rows(df)
   
   # Очистка фич.
   df['BirthDate'] = BirthDate_feature_cleaner(df)
   df['education'] = education_feature_cleaner(df)
   df['employment status'] = employment_status_feature_cleaner(df)
   df['Value'] = Value_feature_cleaner(df)
   df['JobStartDate'] = JobStartDate_feature_cleaner(df)
   return df


def make_features(df:pd.DataFrame)->pd.DataFrame:
   """
   Функция выполняет добавление новых фич.
   """

   df = drop_completely_nan_rows(df)
   
   # Добавление фич.
   df['educationDig_feature_creator'] = educationDig_feature_creator(df)
   df['employment_statusDig_feature_creator'] = employment_statusDig_feature_creator(df)
   df['ValueDig_feature_creator'] = ValueDig_feature_creator(df)
   df['CarierLevel_feature_creator'] = CarierLevel_feature_creator(df)
   df['CarierLevelDig_feature_creator'] = CarierLevelDig_feature_creator(df)
   df['CarierVelocity_feature_creator'] = CarierVelocity_feature_creator(df)
   df[']zodiac'] = zodiac_feature_creator(df)   
   return df

  
