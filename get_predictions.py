from pycaret.classification import load_model, predict_model
import pandas as pd


def get_predictions(df: pd.DataFrame) -> tuple:
    '''Загружаем модели, рассчитываем вероятности.'''
    model_A = load_model('./models/BankA_best')
    pred_A = predict_model(model_A, df, raw_score=True).prediction_score_1

    model_B = load_model('./models/BankB_best')
    pred_B = predict_model(model_B, df, raw_score=True).prediction_score_1

    model_C = load_model('./models/BankC_best')
    pred_C = predict_model(model_C, df, raw_score=True).prediction_score_1

    model_D = load_model('./models/BankD_best')
    pred_D = predict_model(model_D, df, raw_score=True).prediction_score_1

    model_E = load_model('./models/BankE_best')
    pred_E = predict_model(model_E, df, raw_score=True).prediction_score_1

    return (
        float(pred_A),
        float(pred_B),
        float(pred_C),
        float(pred_D),
        float(pred_E)
    )
