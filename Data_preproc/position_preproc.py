import sys
import os

import pandas as pd
import numpy as np

from gensim.models import KeyedVectors
from huggingface_hub import hf_hub_download


model = KeyedVectors.load_word2vec_format(
    hf_hub_download(
        repo_id="Word2vec/nlpl_65",
        filename="model.bin"),
    binary=True,
    unicode_errors="ignore")


def transform_rows(row: str) -> list:
    '''Убираем из строки лишние разделители и разбиваем на слова'''
    try:
        row = row.replace('/', ' ')
        row = row.replace('-', ' ')
        row = row.replace(',', ' ')
        row = row.replace('.', ' ')
        row = row.replace('=', ' ')
        row = row.replace('+', ' ')
        row = row.lower()
        list_of_word = row.split()
    except:
        list_of_word = []

    return list_of_word


def calc_dist(list_of_word: list) -> list:
    '''
    Вычисляем дистанцию между словом "руководитель" и словами в строке.
    В качестве итогового результата оставляем наименьшую дистанцию.
    '''
    list_of_word = list_of_word.copy()
    for idx, word in enumerate(list_of_word):
        try:
            list_of_word[idx] = model.distance(w1="руководитель", w2=word)
            list_of_word[idx] = int(round(list_of_word[idx], 2) * 100)
        except:
            list_of_word[idx] = 100
    try:
        result_dist = min(list_of_word)
    except:
        result_dist = 100
    
    return result_dist


def position_to_dist(df: pd.DataFrame) -> pd.Series:
    '''
    Функция преобразовывает столбец "Position" к числовому виду 
    путем расчета расстояния между эмбедингами слов "руководитель"
    и должности, указанной в данном столбце. Вариант от Ярослава.
    '''
    df = df.copy()
    df.Position = df.Position.apply(transform_rows)
    df.Position = df.Position.apply(calc_dist)

    return df.Position


def CarierLevel_feature_creator(df: pd.DataFrame) -> pd.Series:
    """
    Функция преобразовывает столбец "Position" к категориальному виду
    с приведенным к ограниченному числу уникальных значений содержимым.
    Вариант от Игоря. 
  
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

    df_yaro = df.copy()
    df_yaro['Position'] = position_to_dist(df_yaro)
    path, filename = os.path.split(dataframe_path)
    filename = 'yaro_edited_' + filename
    path = path + filename
    df_yaro.to_csv(path, index=True, sep=';')

    df_igor = df.copy()
    df_igor['Position'] = CarierLevel_feature_creator(df_igor)
    path, filename = os.path.split(dataframe_path)
    filename = 'igor_edited_' + filename
    path = path + filename
    df_igor.to_csv(path, index=True, sep=';')
