import numpy as np
import pandas as pd
import pickle
from fastapi import FastAPI
from fastapi.responses import Response
from typing import Any
from pydantic import BaseModel
import uvicorn
import warnings
warnings.filterwarnings("ignore")

from request_checker import check_n_fill
from nans_filler import fill_nans_pipe
from Yaro_features_creator import Yaro_features_creator_pipe


class Item(BaseModel):
    BirthDate:          Any # 0   BirthDate             datetime64[ns]
    education:          Any # 1   education             object
    employment_status:  Any # 2   employment status     object
    Value:              Any # 3   Value                 object
    JobStartDate:       Any # 4   JobStartDate          datetime64[ns]
    Position:           Any # 5   Position              object
    MonthProfit:        Any # 6   MonthProfit           float64
    MonthExpense:       Any # 7   MonthExpense          float64
    Gender:             Any # 8   Gender                float64
    Family_status:      Any # 9   Family status         object
    ChildCount:         Any # 10  ChildCount            float64
    SNILS:              Any # 11  SNILS                 float64
    Merch_code:         Any # 17  Merch_code            float64
    Loan_amount:        Any # 18  Loan_amount           float64
    Loan_term:          Any # 19  Loan_term             float64
    Goods_category:     Any # 20  Goods_category        object

app = FastAPI()


@app.get("/")
def root():
    '''Get-запрос к корневому каталогу. Возвращает лого команды.'''
    with open("./images/logo.png", "rb") as f:
        img = f.read()

    return Response(content=img, media_type="image/png")


@app.get("/help/")
def find():
    '''Get-запрос для получения описания работы приложения'''
    with open("./images/info.png", "rb") as f:
        img = f.read()

    return Response(content=img, media_type="image/png")


@app.get("/info/")
def find():
    '''Get-запрос для получения описания работы приложения'''
    with open("./images/info.png", "rb") as f:
        img = f.read()

    return Response(content=img, media_type="image/png")


@app.post("/AskOraqul/")
def get_model_prediction(item: Item):
    '''Отправляем предсказание модели по запрошенной строке данных'''
    # Проверяем полученный json и преобразовываем его в pd.DataFrame
    df = check_n_fill(item)
    # Заполнение пропусков
    df = fill_nans_pipe(df)
    # Создаем новые признаки, необходимые для работы модели
    df = Yaro_features_creator_pipe(df)
    #print(df.info())
    # Загружаем модели, рассчитываем вероятности
    model_A = pickle.load(open('./models/BankA_decision.cls', 'rb'))
    pred_A = model_A.predict(df, prediction_type='Probability')[:, -1]
    model_B = pickle.load(open('./models/BankB_decision.cls', 'rb'))
    pred_B = model_B.predict(df, prediction_type='Probability')[:, -1]
    model_C = pickle.load(open('./models/BankC_decision.cls', 'rb'))
    pred_C = model_C.predict(df, prediction_type='Probability')[:, -1]
    model_D = pickle.load(open('./models/BankD_decision.cls', 'rb'))
    pred_D = model_D.predict(df, prediction_type='Probability')[:, -1]
    model_E = pickle.load(open('./models/BankE_decision.cls', 'rb'))
    pred_E = model_E.predict(df, prediction_type='Probability')[:, -1]

    return (
        float(pred_A),
        float(pred_B),
        float(pred_C),
        float(pred_D),
        float(pred_E)
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000, log_level="info")
    #uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")