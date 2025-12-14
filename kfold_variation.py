# 1 Para manipular datos
import pandas as pd        # Leer CSV, manipular DataFrames
import numpy as np         # Operaciones numéricas y matrices

# 2 Para separar train/test y otras utilidades de ML
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error  # Métricas de evaluación

# 3 LightGBM
import lightgbm as lgb

# 4 Para visualizaciones (no usado aquí, pero útil)
import os
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

# Cargar datos
data = pd.read_csv("cavalli_statistical.csv", sep=",")
data3 = pd.read_csv("augmented_metadata.csv", sep=",")


print(len(data3)) # para verificar el número de filas
print(data3.index)  # para ver el rango de índices exacto

data3.drop(columns=["Unnamed: 0"], inplace=True)
data.drop(columns=["Unnamed: 0"], inplace=True)
data2 = data3.iloc[:763]  # toma desde la fila 0 hasta la 764 inclusiveç, puesto que otro archivo tiene mas filas y esta limitado por el más pequeño




#label enconding 
map = {'WNT':0,'SHH':1,'Group 3':2,'Group 4':3,'G3-G4':4} 
data2 = data2['0'].map(map)

#esto es para verificar que las cantidad de forma visual filas de X e y coinciden
print("Filas X:", data.shape[0])
print("Filas y:", data2.shape[0])


# Eliminar filas con etiquetas faltantes
mask = data2.notna()
data_clean = data[mask]
labels_clean = data2[mask]

from sklearn.model_selection import StratifiedKFold

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_index, test_index in kf.split(data, data2):
    x_train, x_test = data.iloc[train_index], data.iloc[test_index]
    y_train, y_test = data2.iloc[train_index], data2.iloc[test_index]  # usar labels codificadas


    # Crear conjuntos de datos LightGBM
    train = lgb.Dataset(x_train, y_train)
    eval = lgb.Dataset(x_test, y_test, reference=train)
    # Configurar parámetros del modelo LightGBM
    params= {
        'objective': 'multiclass',
        'num_class': 5,
        'metric': 'multi_logloss',
        'learning_rate': 0.10,
        'num_leaves': 100,
        'max_depth': 20,
        'min_data_in_leaf': 30,
        'feature_fraction': 0.8,
        'bagging_fraction': 1,
        'bagging_freq': 1,
        'lambda_l1': 3,
        'lambda_l2': 10,
        'verbosity': -1
    }



    # Verificar si hay valores NaN en y_test
    print(y_test.isna().sum())

    # Entrenar el modelo LightGBM
    gbm = lgb.train(
        params,
        train,
        valid_sets=[eval],
        num_boost_round=3000, 
        callbacks=[lgb.early_stopping(50)]  
    )

    # Hacer predicciones
    y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration)

    # Evaluar el modelo
    from sklearn.metrics import confusion_matrix, log_loss
    y_pred_labels = np.argmax(y_pred, axis=1)
    print(confusion_matrix(y_test, y_pred_labels))
    print(log_loss(y_test, y_pred))
    print(x_test.shape)









