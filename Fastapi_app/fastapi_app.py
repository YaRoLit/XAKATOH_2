import numpy as np
import pandas as pd

from fastapi import FastAPI
from fastapi.responses import Response, FileResponse
from pydantic import BaseModel
import uvicorn

from request_checker import check_n_fill, fill_nans


class Item(BaseModel):
    BirthDate:          str # 0   BirthDate             datetime64[ns]
    education:          str # 1   education             object
    employment_status:  str # 2   employment status     object
    Value:              str # 3   Value                 object
    JobStartDate:       str # 4   JobStartDate          datetime64[ns]
    Position:           str # 5   Position              object
    MonthProfit:        str # 6   MonthProfit           float64
    MonthExpense:       str # 7   MonthExpense          float64
    Gender:             str # 8   Gender                float64
    Family_status:      str # 9   Family status         object
    ChildCount:         str # 10  ChildCount            float64
    SNILS:              str # 11  SNILS                 float64
    Merch_code:         str # 17  Merch_code            float64
    Loan_amount:        str # 18  Loan_amount           float64
    Loan_term:          str # 19  Loan_term             float64
    Goods_category:     str # 20  Goods_category        object

app = FastAPI()


@app.get("/")
def root():
    '''Get-запрос к корневому каталогу. Возвращает лого команды.'''
    with open("logo.png", "rb") as f:
        img = f.read()

    return Response(content=img, media_type="image/png")


@app.get("/help/")
def find():
    '''Get-запрос для получения описания работы приложения'''
    with open("info.png", "rb") as f:
        img = f.read()

    return Response(content=img, media_type="image/png")


@app.get("/info/")
def find():
    '''Get-запрос для получения описания работы приложения'''
    with open("info.png", "rb") as f:
        img = f.read()

    return Response(content=img, media_type="image/png")


@app.post("/AskOraqul/")
def get_model_prediction(item: Item):
    '''Отправляем предсказание модели по запрошенной строке данных'''
    df = check_n_fill(item)
    for col in df.columns:
        print(col, '\t'*3, df.loc[0, col])
    if df.isna().sum().sum():
        df = fill_nans(df)
    # Тут нужно будет много написать

    return {"BankA_decision": "OK"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000, log_level="info")