from fastapi import FastAPI
from fastapi.responses import Response
from typing import Any
from pydantic import BaseModel
import uvicorn
from request_checker import check_n_fill
from nans_filler import fill_nans_pipe
from features_creator_by_Yaro import features_creator_pipe
from get_predictions import get_predictions
import warnings
warnings.filterwarnings("ignore")

'''
Основной скрипт работы сервера fastapi.

Структура данных, которые предполагается получить для обеспечения
корректной работы моделей, приведена ниже и описана в классе Item.
Она определена исходя из структуры набора данных, переданных
заказчиком изначально для решения задачи.

Все именнованные поля могут получать от json из post запроса тип данных
Any. Это сделано для того, чтобы уменьшить количество ошибок при конвертации
данных на этапе обработчика post запроса fastapi. В дальнейшем перепроверка
соответствия данных производится в модуле request_checker, нём же описан
принцип его работы.

После проверки и предварительного преобразования данных производится
четырехступенчатое заполнение пропусков в полученном датафрейме, реализованное
в конвейере в модуле nans_filler, где приведено подробное описание этапов
его работы.

Заключительным этапом предварительной подготовки данных является конвейер
обогащения датафрейма новыми признаками, который реализован в модуле
features_creator. После этого датафрейм со всеми необходимыми параметрами
передается на вход предобученных моделей, с помощью которых производится
расчёт вероятности одобрения кредита по каждому из исследуемых банков.
Пять полученных значений вероятностей возвращаются в виде кортежа как
ответ на post запрос.
'''


class Item(BaseModel):          # ожидаемое значение
    BirthDate:          Any     # 0 datetime64[ns]
    education:          Any     # 1 object
    employment_status:  Any     # 2 object
    Value:              Any     # 3 object
    JobStartDate:       Any     # 4 datetime64[ns]
    Position:           Any     # 5 object
    MonthProfit:        Any     # 6 float64
    MonthExpense:       Any     # 7 float64
    Gender:             Any     # 8 float64
    Family_status:      Any     # 9 object
    ChildCount:         Any     # 10 float64
    SNILS:              Any     # 11 float64
    Merch_code:         Any     # 17 float64
    Loan_amount:        Any     # 18 float64
    Loan_term:          Any     # 19 float64
    Goods_category:     Any     # 20 object


app = FastAPI()


@app.get("/")
def root():
    '''Get-запрос к корневому каталогу. Возвращает лого команды.'''
    with open("./images/logo.png", "rb") as f:
        img = f.read()

    return Response(content=img, media_type="image/png")


@app.get("/help/")
def show_help():
    '''
    Get-запрос для получения описания работы приложения. Возвращает
    информационный слайд.
    '''
    with open("./images/info.png", "rb") as f:
        img = f.read()

    return Response(content=img, media_type="image/png")


@app.get("/info/")
def show_info():
    '''Get-запрос для получения описания работы приложения'''
    with open("./images/info.png", "rb") as f:
        img = f.read()

    return Response(content=img, media_type="image/png")


@app.post("/AskOraqul/")
def get_model_prediction(item: Item):
    '''Отправляем предсказание модели по запрошенной строке данных'''
    # Проверяем полученный json и преобразовываем его в pd.DataFrame
    df = check_n_fill(item)
    # Проверяем наличие пропусков, заполняем их с помощью разных стратегий
    df = fill_nans_pipe(df)
    # Обогащаем датасет новыми признаками, необходимыми для работы модели
    df = features_creator_pipe(df)
    # Загружаем модели, рассчитываем вероятности
    # Теоретически, модели можно загружать сразу при объявлении глобальных
    # переменных скрипта, а не в теле фунции. При этом увеличится
    # быстродействие скрипта, так как модели будут в оперативной памяти.
    # Но при приведенном ниже варианте загрузки файлы моделей можно поменять
    # "на горячую", не останавливая работу скрипта, а просто перезаписав
    # их в соответствующей папке.
    return get_predictions(df) 



if __name__ == "__main__":
    # Для развертывания приложения на виртуальной машине в Интернет
    # uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")
    # Для развертывания приложения в локальной сети
    uvicorn.run(app, host="127.0.0.1", port=5000, log_level="info")
