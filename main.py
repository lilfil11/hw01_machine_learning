from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict

import pickle
import numpy as np
import pandas as pd


app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


model = pickle.load(open('model.pickle', 'rb'))
normalizer = pickle.load(open('normalizer.pickle', 'rb'))
encoder = pickle.load(open('encoder.pickle', 'rb'))
mis_replacer = pickle.load(open('mis_replacer.pickle', 'rb'))


def prepare_data(data, num):
    # переводим json в датафрейм
    if num == 'one':
        data = dict(data)
        df = pd.DataFrame([data])
    else:
        for i in range(len(data)):
             data[i] = dict(data[i])
        df = pd.DataFrame(data)

    # приведём некоторые столбцы к float
    df['mileage'] = df['mileage'].str.replace(r'[^\.0-9]+', '', regex=True).astype('float')
    df['engine'] = df['engine'].str.replace(r'[^\.0-9]+', '', regex=True).astype('float')
    df['max_power'] = df['max_power'].str.replace(r'[^\.0-9]+', '', regex=True).astype('float')

    # удаляем столбец
    df = df.drop('torque', axis=1)

    # заполняем пропуски в числовых признаках
    cat_features_mask = (df.dtypes == "object").values
    df_cat = df[df.columns[cat_features_mask]]
    df_real = df[df.columns[~cat_features_mask]]
    df_real = pd.DataFrame(data=mis_replacer.transform(df_real), columns=df_real.columns)
    df = pd.concat([df_cat, df_real], axis=1)

    # приводим некоторые признаки к int
    df['engine'] = df['engine'].astype('int')
    df['seats'] = df['seats'].astype('int')

    # работаем с признаками
    df = df.drop(columns=['selling_price', 'name'])
    df['seats'] = df['seats'].astype('str')
    df['year_squared'] = df['year'] ** 2

    real_columns = ['year', 'km_driven', 'mileage', 'engine', 'max_power']
    for i in range(5):
        for j in range(i, 5):
            if (i == 0) & (j == 0):
                continue
            df[real_columns[i] + '_' + real_columns[j]] = df[real_columns[i]] * df[real_columns[j]]

    for i in range(5):
        if (i == 1) | (i == 2) | (i == 4):
            continue
        df['km_driven/' + real_columns[i]] = df['km_driven'] / df[real_columns[i]]

    df['ind_km_mean'] = (df['seller_type'] == 'Individual') & (df['km_driven'] > df['km_driven'].mean())
    df['owner_three'] = (df['owner'] != 'First Owner') & (df['owner'] != 'Second Owner')

    miss_features = ['engine', 'mileage', 'max_power']
    for i in range(3):
        df[miss_features[i] + '_miss'] = (df[miss_features[i]] == df[miss_features[i]].mean())

    df['year_log'] = np.log(df['year'])
    df['engine_log'] = np.log(df['engine'])
    df['km_driven_log'] = np.log(df['km_driven'])

    cat_features_mask = (df.dtypes == "object").values
    df_cat = df[df.columns[cat_features_mask]]
    df = pd.concat([df.drop(columns=df_cat.columns),
                         pd.DataFrame(encoder.transform(df_cat).toarray(),
                                      columns=encoder.get_feature_names_out())], axis=1)

    df = pd.DataFrame(data=normalizer.transform(df), columns=df.columns)

    return df


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    X = prepare_data(item, 'one')
    y = np.exp(model.predict(X))
    return y[0][0]


@app.post("/predict_items")
def predict_items(items: List[Item]):
    X = prepare_data(items, 'non-num')
    X['predict'] = np.exp(model.predict(X))
    return X.to_dict()