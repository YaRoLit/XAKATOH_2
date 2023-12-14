import pickle
import pandas as pd


def get_predictions(df: pd.DataFrame) -> tuple:
    '''Загружаем модели, рассчитываем вероятности.'''
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
