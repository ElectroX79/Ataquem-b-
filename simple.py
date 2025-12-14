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



# Dividir en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(
    data, data2, test_size=0.2, shuffle=True, random_state=42
)

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










import shap # Para interpretabilidad y explicabilidad del modelo LightGBM
pd.set_option('display.max_rows', 120)



# Obtener los valores SHAP para el conjunto de test
explainer = shap.TreeExplainer(gbm)
shap_values = explainer.shap_values(x_test)

# Verificar el formato de shap_values y convertir si es necesario
if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
    # Convertir de (n_samples, n_features, n_classes) a lista de (n_samples, n_features)
    shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]

# Obtener los nombres de las columnas
feature_names = x_test.columns.tolist()

# Para cada clase, calcular la importancia media absoluta de cada feature
for class_idx in range(5):
    class_names = {0: 'WNT', 1: 'SHH', 2: 'Group 3', 3: 'Group 4', 4: 'G3-G4'}
    
    print(f"\n{'='*80}")
    print(f"Top 100 genes más importantes para clase {class_idx} ({class_names[class_idx]})")
    print(f"{'='*80}\n")
    
    # Calcular la importancia media absoluta para esta clase
    mean_abs_shap = np.abs(shap_values[class_idx]).mean(axis=0)
    
    # Crear DataFrame con genes e importancias
    importance_df = pd.DataFrame({
        'Gene': feature_names,
        'Mean_Abs_SHAP': mean_abs_shap
    })
    
    # Ordenar por importancia descendente y tomar top 100
    top_100 = importance_df.sort_values('Mean_Abs_SHAP', ascending=False).head(500)
    
    # Resetear el índice para mejor visualización
    top_100_display = top_100.reset_index(drop=True)
    top_100_display.index = top_100_display.index + 1  # Empezar desde 1
    
    print(top_100_display.to_string())
    print(f"\n")

#Guardar los resultados en archivos CSV
print("\nGuardando resultados en archivos CSV...")

for class_idx in range(5):
    class_names = {0: 'WNT', 1: 'SHH', 2: 'Group 3', 3: 'Group 4', 4: 'G3-G4'}
    mean_abs_shap = np.abs(shap_values[class_idx]).mean(axis=0)
    importance_df = pd.DataFrame({
        'Gene': feature_names,
        'Mean_Abs_SHAP': mean_abs_shap
    })
    
    top_100 = importance_df.sort_values('Mean_Abs_SHAP', ascending=False).head(500)
    
    filename = f"top_500_genes_class_{class_idx}_{class_names[class_idx].replace(' ', '_')}.csv"
    top_100.to_csv(filename, index=False)
    print(f"Guardado: {filename}")

print("\n Proceso completado")




##fuera de uso de shap
##fuera de uso de shap
##fuera de uso de shap
##fuera de uso de shap-----------------------------------------------------------

# Calcular puntuaciones normalizadas para Group 3 y Group 4 
datatop_g3 = pd.read_csv("top_500_genes_class_2_Group_3.csv", sep=",")
datatop_g4 = pd.read_csv("top_500_genes_class_3_Group_4.csv", sep=",")


# Eliminar columnas con todos los valores iguales a cero
datatop_g3 = datatop_g3.loc[:, (datatop_g3 != 0).any(axis=0)]
datatop_g4 = datatop_g4.loc[:, (datatop_g4 != 0).any(axis=0)]

# Normalizar las importancias
datatop_g3['Mean_Abs_SHAP'] = datatop_g3['Mean_Abs_SHAP'] / datatop_g3['Mean_Abs_SHAP'].sum()
datatop_g4['Mean_Abs_SHAP'] = datatop_g4['Mean_Abs_SHAP'] / datatop_g4['Mean_Abs_SHAP'].sum()




data_in=pd.read_csv("cavalli_statistical.csv", sep=",")

# Indexar los DataFrames por 'Gene' para facilitar la selección y mejorar el rendimiento
datatop_g3_indexed = datatop_g3.set_index('Gene')
datatop_g4_indexed = datatop_g4.set_index('Gene')

# Encontrar genes comunes entre los DataFrames y data_in
common_genes_g3 = datatop_g3_indexed.index.intersection(data_in.columns)
common_genes_g4 = datatop_g4_indexed.index.intersection(data_in.columns)

# Weighted sum para Group 3
weighted_sum_g3 = (data_in[common_genes_g3] * datatop_g3_indexed.loc[common_genes_g3, 'Mean_Abs_SHAP']).sum(axis=1)

# Weighted sum para Group 4
weighted_sum_g4 = (data_in[common_genes_g4] * datatop_g4_indexed.loc[common_genes_g4, 'Mean_Abs_SHAP']).sum(axis=1)

total = weighted_sum_g3 + weighted_sum_g4

weighted_sum_g3_norm = weighted_sum_g3 / total
weighted_sum_g4_norm = weighted_sum_g4 / total

# Imprimir
print("Group 3 (norm):")
print(weighted_sum_g3_norm)

print("\nGroup 4 (norm):")
print(weighted_sum_g4_norm)


