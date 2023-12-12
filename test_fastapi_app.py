import pandas as pd
import numpy as np
import fastapi_app as api_app
from fastapi.testclient import TestClient
import sys, asyncio


'''
Интеграционные тесты для проверки работоспособности приложения fastapi.
Внедрены в Workflow репозитория проекта на Github, используются для
проверки всех поступающий pull-requests, чтобы никто не свалил приложение.
Тесты проверяют работу приложения полностью, для этого разворачивают
тестовый сервер fastapi, на который подается множество запросов
с параметрами реальных клиентов из подгружаемого датасета, но
зашумленных np.NaNs. При этом проверяется устойчивость работы
приложения в части очистки поступающих данных и расчета вероятностей
моделями (возвращаемое ими значение также проверяется по принципу
"является числом от 0 до 1 включительно"). Также несколько простых тестов
обращения к корневой и информационным страницам (проверяю возврат
соответствующих картинок).
'''


if sys.platform == "win32" and (3, 8, 0) <= sys.version_info < (3, 9, 0):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

client = TestClient(api_app.app)

with open("./images/logo.png", "rb") as f:
    logo = f.read()

with open("./images/info.png", "rb") as f:
    info = f.read()


def test_root():
    response = client.get('/')
    assert response.status_code == 200
    assert response.content == logo


def test_info():
    response = client.get('/info/')
    assert response.status_code == 200
    assert response.content == info


def test_help():
    response = client.get('/info/')
    assert response.status_code == 200
    assert response.content == info


def test_oraqul():
    df = pd.read_csv(
        filepath_or_buffer='./data/api_test_df.csv',
        sep=';',
        index_col='SkillFactory_Id'
        )
    banks = [
        'BankA_decision',
        'BankB_decision',
        'BankC_decision',
        'BankD_decision',
        'BankE_decision'
        ]
    df = df.drop(banks, axis='columns')
    df = df.dropna(how='all')
    # Здесь зашумляем датасет случайными пропусками
    for col in df.columns:
        sample_index =  df.sample(round(df.shape[0] * 0.1)).index
        df.loc[sample_index, col] = np.NaN
    df.rename(
        columns={"employment status": "employment_status",
                 "Family status": "Family_status"}, 
        inplace=True
        )
    frame_for_send = df.sample(400).to_dict('records')
    for i in range(len(frame_for_send)):
        json_data = df.iloc[i].to_json().encode('utf8')
        response = client.post('/AskOraqul/', data=json_data)
        assert response.status_code == 200
        banks_decisions = response.content.decode()[1:-1].split(',')
        assert 0 <= float(banks_decisions[0]) <= 1
        assert 0 <= float(banks_decisions[1]) <= 1
        assert 0 <= float(banks_decisions[2]) <= 1
        assert 0 <= float(banks_decisions[3]) <= 1
        assert 0 <= float(banks_decisions[4]) <= 1