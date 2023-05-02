import pandas as pd
import numpy as np
import joblib as jb
import tensorflow as tf

# DeepL model
class DNN_():

    def __init__(self, name):
        self.name = name
        self.scales = {}
        self.models = {}
        for z in ['A','B','C']:
            self.scales[z] = jb.load(f'./models/scaler_dnn{z}.joblib')
            self.models[z] = tf.keras.saving.load_model(f'./models/dnn{z}.keras')

    def predict_one(self, df):
        Xt = df.copy().drop(columns='RainTomorrow').reset_index(drop=True)
        kc = Xt.KCode.str[0].at[0]

        dt = Xt['Date']
        Xt = Xt[['MinTemp','MaxTemp','Rainfall','Evaporation','Sunshine',
                'WindGustSpeed','WindSpeed9am','WindSpeed3pm','Humidity9am',
                'Humidity3pm','Pressure9am','Pressure3pm','Cloud9am','Cloud3pm',
                'Temp9am','Temp3pm']].fillna(0)

        scale = self.scales[kc]
        model = self.models[kc]
        Xt = scale.transform(Xt)
        pred = model.predict(Xt)
        pred = np.where(pred >= 0.5, 'Yes', 'No').flatten()[0]

        return {'Date': dt, 'Pred': pred, 'Real': None, 'Good': None}

    def predict(self, df):
        Xt = df.copy().dropna(subset='RainTomorrow').reset_index(drop=True)
        kc = Xt.KCode.str[0].at[0]

        dt = Xt.Date
        yt = Xt.pop('RainTomorrow')
        Xt = Xt[['MinTemp','MaxTemp','Rainfall','Evaporation','Sunshine',
                'WindGustSpeed','WindSpeed9am','WindSpeed3pm','Humidity9am',
                'Humidity3pm','Pressure9am','Pressure3pm','Cloud9am','Cloud3pm',
                'Temp9am','Temp3pm']].fillna(0)

        scale = self.scales[kc]
        model = self.models[kc]
        Xt = scale.transform(Xt)
        pred = model.predict(Xt)
        pred = np.where(pred >= 0.5, 'Yes', 'No').flatten()

        return pd.DataFrame({'Date': dt, 'Pred': pred, 'Real': yt, 'Good': (pred == yt).astype(int)})

models = [
    DNN_('Dense Neural Net')
]

def acc_yes_nop(res):
    """Compute metrics.
    
    Parameters
    ----------
    res: pandas.DataFrame
        Data to compute metrics for, as given by model(s) prediction.

    Returns
    -------
    A tuple of 3 elements: accuracy, Yes an No rates
    """
    acc = (res.Good.sum() / len(res))
    yer = len(res[res.Real == 'Yes'])
    yes = len(res[res.Pred == 'Yes'])
    nor = len(res[res.Real == 'No'])
    nop = len(res[res.Pred == 'No'])
    yes = len(res[(res.Pred == 'Yes') & (res.Good == 1)]) / yer if yer > 0 else \
        0 if yes > 0 else 1
    nop = len(res[(res.Pred == 'No') & (res.Good == 1)]) / nor if nor > 0 else \
        0 if nop > 0 else 1
    return acc * 100, yes * 100, nop * 100