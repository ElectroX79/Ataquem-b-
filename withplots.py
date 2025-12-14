# 1 Para manipular datos
import pandas as pd
import numpy as np

# 2 Para separar train/test y otras utilidades de ML
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, confusion_matrix

# 3 LightGBM
import lightgbm as lgb

# 4 Para visualizaciones
import os
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

# 5 Para UMAP y visualización interactiva
import umap
import plotly.express as px
from datetime import datetime

# 6 Para SHAP
import shap

# ============================================================================
# FUNCIONES DE VISUALIZACIÓN UMAP
# ============================================================================

def plot_umap_binary(
    data,
    clinical,
    colors_dict,
    shapes_dict=None,
    n_components=2,
    save_fig=False,
    save_as=None,
    seed=None,
    title='UMAP',
    show=True,
    marker_size=8,
):
    """
    Plot UMAP con colores discretos para diferentes grupos.
    """
    # Check number of samples is the first dimension of data:
    if data.shape[0] != clinical.shape[0]:
        data = data.T
        if data.shape[0] != clinical.shape[0]:
            raise ValueError("Data and clinical metadata must have the same number of samples")

    if n_components not in (2, 3):
        raise ValueError("n_components must be 2 or 3 for plot_umap_plotly")

    today = datetime.now().strftime("%Y%m%d")
    if save_as is None:
        suffix = "UMAP" if n_components == 2 else "3D_UMAP"
        save_as = f"{today}_{suffix}"

    if seed is not None:
        umap_ = umap.UMAP(n_components=n_components, random_state=seed)
    else:
        umap_ = umap.UMAP(n_components=n_components)

    # data: samples x features
    X_umap = umap_.fit_transform(data)
    print("X_umap.shape", X_umap.shape)

    # Determine color and shape series from clinical
    if isinstance(clinical, pd.DataFrame):
        color_col = clinical.columns[0]
        color_series = clinical[color_col]
        if shapes_dict is not None and clinical.shape[1] >= 2:
            shape_col = clinical.columns[1]
            shape_series = clinical[shape_col]
        else:
            shape_series = None
    elif isinstance(clinical, pd.Series):
        color_series = clinical
        shape_series = None
    else:
        raise ValueError("clinical must be a pandas Series or DataFrame")

    # Build plotting DataFrame
    all_patients = data.index.tolist()
    df_plot = pd.DataFrame(
        {
            "sample": all_patients,
            "group": color_series.loc[all_patients].values,
            "UMAP_1": X_umap[:, 0],
            "UMAP_2": X_umap[:, 1],
        }
    )
    if n_components == 3:
        df_plot["UMAP_3"] = X_umap[:, 2]

    if shape_series is not None:
        df_plot["shape"] = shape_series.loc[all_patients].values

    # Build color sequence
    unique_groups = df_plot["group"].unique()
    color_sequence = [colors_dict[g] for g in unique_groups]

    # Prepare symbol mapping if shapes are used
    symbol_map = None
    if "shape" in df_plot.columns and shapes_dict is not None:
        matplot_to_plotly = {
            'o': 'circle', 's': 'square', '^': 'triangle-up', 'v': 'triangle-down',
            'D': 'diamond', 'd': 'diamond-wide', 'X': 'x', 'x': 'x', '*': 'star',
            '+': 'cross', 'p': 'pentagon', 'h': 'hexagon', 'H': 'hexagon2'
        }
        unique_shapes = df_plot["shape"].unique()
        symbol_map = {}
        for sh in unique_shapes:
            marker = shapes_dict.get(sh, shapes_dict.get(str(sh), sh))
            symbol = matplot_to_plotly.get(marker, marker)
            symbol_map[sh] = symbol

    # Create plotly figure
    if n_components == 2:
        fig = px.scatter(
            df_plot,
            x="UMAP_1",
            y="UMAP_2",
            color="group",
            color_discrete_sequence=color_sequence,
            hover_name="sample",
            template="simple_white",
            width=800,
            height=800,
            symbol="shape" if "shape" in df_plot.columns and symbol_map is not None else None,
            symbol_map=symbol_map if symbol_map is not None else None,
        )
        fig.update_layout(
            title=title,
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
        )
        fig.update_traces(marker=dict(size=marker_size))
    else:
        fig = px.scatter_3d(
            df_plot,
            x="UMAP_1",
            y="UMAP_2",
            z="UMAP_3",
            color="group",
            color_discrete_sequence=color_sequence,
            hover_name="sample",
            template="simple_white",
            width=800,
            height=800,
            symbol="shape" if "shape" in df_plot.columns and symbol_map is not None else None,
            symbol_map=symbol_map if symbol_map is not None else None,
        )
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="UMAP 1",
                yaxis_title="UMAP 2",
                zaxis_title="UMAP 3",
            ),
        )
        fig.update_traces(marker=dict(size=marker_size))

    # Optional saving
    if save_fig:
        base_dir = os.path.dirname(save_as)
        if base_dir and not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)
        fig.write_html(f"{save_as}.html")
        for extension in ['png', 'pdf', 'svg']:
            print(f"Saved UMAP plotly figure to: {save_as}.{extension}")
            try:
                fig.write_image(f"{save_as}.{extension}", scale=2)
            except Exception as e:
                print(f"Could not save {extension}: {e}")
    
    if show:
        fig.show()
    
    return fig


def plot_umap_spectrum(
    data,
    clinical,
    colormap='viridis',
    shapes_dict=None,
    n_components=2,
    save_fig=False,
    save_as=None,
    seed=None,
    title='UMAP',
    show=True,
    marker_size=8,
    color_range=None,
    colorbar_title=None,
):
    """
    Plot UMAP con escala de color continua para valores numéricos.
    """
    # Check number of samples is the first dimension of data:
    if data.shape[0] != clinical.shape[0]:
        data = data.T
        if data.shape[0] != clinical.shape[0]:
            raise ValueError("Data and clinical metadata must have the same number of samples")

    if n_components not in (2, 3):
        raise ValueError("n_components must be 2 or 3 for plot_umap_spectrum")

    today = datetime.now().strftime("%Y%m%d")
    if save_as is None:
        suffix = "UMAP_spectrum" if n_components == 2 else "3D_UMAP_spectrum"
        save_as = f"{today}_{suffix}"

    if seed is not None:
        umap_ = umap.UMAP(n_components=n_components, random_state=seed)
    else:
        umap_ = umap.UMAP(n_components=n_components)

    # data: samples x features
    X_umap = umap_.fit_transform(data)
    print("X_umap.shape", X_umap.shape)

    # Determine color and shape series from clinical
    if isinstance(clinical, pd.DataFrame):
        color_col = clinical.columns[0]
        color_series = clinical[color_col]
        if shapes_dict is not None and clinical.shape[1] >= 2:
            shape_col = clinical.columns[1]
            shape_series = clinical[shape_col]
        else:
            shape_series = None
    elif isinstance(clinical, pd.Series):
        color_series = clinical
        shape_series = None
    else:
        raise ValueError("clinical must be a pandas Series or DataFrame")

    print("color_series.shape", color_series.shape)
    print(f"color_series range: [{color_series.min():.3f}, {color_series.max():.3f}]")

    # Validate numeric values
    if not pd.api.types.is_numeric_dtype(color_series):
        raise ValueError("For continuous coloring, clinical data must contain numeric values")

    # Build plotting DataFrame
    all_patients = data.index.tolist()
    df_plot = pd.DataFrame(
        {
            "sample": all_patients,
            "color_value": color_series.loc[all_patients].values,
            "UMAP_1": X_umap[:, 0],
            "UMAP_2": X_umap[:, 1],
        }
    )
    if n_components == 3:
        df_plot["UMAP_3"] = X_umap[:, 2]

    if shape_series is not None:
        df_plot["shape"] = shape_series.loc[all_patients].values

    # Set up color range
    if color_range is None:
        color_range = [df_plot["color_value"].min(), df_plot["color_value"].max()]

    # Set colorbar title
    if colorbar_title is None:
        if isinstance(clinical, pd.Series):
            colorbar_title = clinical.name if clinical.name else "Value"
        else:
            colorbar_title = color_series.name if color_series.name else "Value"

    # Prepare symbol mapping if shapes are used
    symbol_map = None
    if "shape" in df_plot.columns and shapes_dict is not None:
        matplot_to_plotly = {
            'o': 'circle', 's': 'square', '^': 'triangle-up', 'v': 'triangle-down',
            'D': 'diamond', 'd': 'diamond-wide', 'X': 'x', 'x': 'x', '*': 'star',
            '+': 'cross', 'p': 'pentagon', 'h': 'hexagon', 'H': 'hexagon2'
        }
        unique_shapes = df_plot["shape"].unique()
        symbol_map = {}
        for sh in unique_shapes:
            marker = shapes_dict.get(sh, shapes_dict.get(str(sh), sh))
            symbol = matplot_to_plotly.get(marker, marker)
            symbol_map[sh] = symbol

    # Create plotly figure with continuous color scale
    if n_components == 2:
        fig = px.scatter(
            df_plot,
            x="UMAP_1",
            y="UMAP_2",
            color="color_value",
            color_continuous_scale=colormap,
            hover_name="sample",
            hover_data={"color_value": ":.3f"},
            template="simple_white",
            width=800,
            height=800,
            range_color=color_range,
            symbol="shape" if "shape" in df_plot.columns and symbol_map is not None else None,
            symbol_map=symbol_map if symbol_map is not None else None,
        )
        fig.update_layout(
            title=title,
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            coloraxis_colorbar=dict(title=colorbar_title)
        )
        fig.update_traces(marker=dict(size=marker_size))
    else:
        fig = px.scatter_3d(
            df_plot,
            x="UMAP_1",
            y="UMAP_2",
            z="UMAP_3",
            color="color_value",
            color_continuous_scale=colormap,
            hover_name="sample",
            hover_data={"color_value": ":.3f"},
            template="simple_white",
            width=800,
            height=800,
            range_color=color_range,
            symbol="shape" if "shape" in df_plot.columns and symbol_map is not None else None,
            symbol_map=symbol_map if symbol_map is not None else None,
        )
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="UMAP 1",
                yaxis_title="UMAP 2",
                zaxis_title="UMAP 3",
            ),
            coloraxis_colorbar=dict(title=colorbar_title)
        )
        fig.update_traces(marker=dict(size=marker_size))

    # Optional saving
    if save_fig:
        base_dir = os.path.dirname(save_as)
        if base_dir and not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)
        fig.write_html(f"{save_as}.html")
        for extension in ['png', 'pdf', 'svg']:
            print(f"Saved UMAP spectrum figure to: {save_as}.{extension}")
            try:
                fig.write_image(f"{save_as}.{extension}", scale=2)
            except Exception as e:
                print(f"Could not save {extension}: {e}")
    
    if show:
        fig.show()
    
    return fig


# ============================================================================
# CÓDIGO PRINCIPAL
# ============================================================================

# Cargar datos
data = pd.read_csv("cavalli_statistical.csv", sep=",")
data3 = pd.read_csv("augmented_metadata.csv", sep=",")

print(len(data3))
print(data3.index)

data3.drop(columns=["Unnamed: 0"], inplace=True)
data.drop(columns=["Unnamed: 0"], inplace=True)
data2 = data3.iloc[:763]

# Label encoding
map_labels = {'WNT':0, 'SHH':1, 'Group 3':2, 'Group 4':3, 'G3-G4':4}
data2 = data2['0'].map(map_labels)

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
params = {
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
y_pred_labels = np.argmax(y_pred, axis=1)

# Evaluar el modelo
print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred_labels))
print(f"\nLog Loss: {log_loss(y_test, y_pred):.4f}")

# ============================================================================
# ANÁLISIS SHAP
# ============================================================================

pd.set_option('display.max_rows', 120)

# Obtener los valores SHAP para el conjunto de test
explainer = shap.TreeExplainer(gbm)
shap_values = explainer.shap_values(x_test)

# Verificar el formato de shap_values y convertir si es necesario
if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
    shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]

# Obtener los nombres de las columnas
feature_names = x_test.columns.tolist()

# Para cada clase, calcular la importancia media absoluta de cada feature
for class_idx in range(5):
    class_names = {0: 'WNT', 1: 'SHH', 2: 'Group 3', 3: 'Group 4', 4: 'G3-G4'}
    
    print(f"\n{'='*80}")
    print(f"Top 100 genes más importantes para clase {class_idx} ({class_names[class_idx]})")
    print(f"{'='*80}\n")
    
    mean_abs_shap = np.abs(shap_values[class_idx]).mean(axis=0)
    
    importance_df = pd.DataFrame({
        'Gene': feature_names,
        'Mean_Abs_SHAP': mean_abs_shap
    })
    
    top_100 = importance_df.sort_values('Mean_Abs_SHAP', ascending=False).head(500)
    top_100_display = top_100.reset_index(drop=True)
    top_100_display.index = top_100_display.index + 1
    
    print(top_100_display.to_string())
    print(f"\n")

# Guardar los resultados en archivos CSV
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

print("\nProceso completado")

# ============================================================================
# CÁLCULO DE PUNTUACIONES NORMALIZADAS
# ============================================================================

datatop_g3 = pd.read_csv("top_500_genes_class_2_Group_3.csv", sep=",")
datatop_g4 = pd.read_csv("top_500_genes_class_3_Group_4.csv", sep=",")

# Eliminar columnas con todos los valores iguales a cero
datatop_g3 = datatop_g3.loc[:, (datatop_g3 != 0).any(axis=0)]
datatop_g4 = datatop_g4.loc[:, (datatop_g4 != 0).any(axis=0)]

# Normalizar las importancias
datatop_g3['Mean_Abs_SHAP'] = datatop_g3['Mean_Abs_SHAP'] / datatop_g3['Mean_Abs_SHAP'].sum()
datatop_g4['Mean_Abs_SHAP'] = datatop_g4['Mean_Abs_SHAP'] / datatop_g4['Mean_Abs_SHAP'].sum()

data_in = pd.read_csv("cavalli_statistical.csv", sep=",")

# Indexar los DataFrames por 'Gene'
datatop_g3_indexed = datatop_g3.set_index('Gene')
datatop_g4_indexed = datatop_g4.set_index('Gene')

# Encontrar genes comunes
common_genes_g3 = datatop_g3_indexed.index.intersection(data_in.columns)
common_genes_g4 = datatop_g4_indexed.index.intersection(data_in.columns)

# Weighted sum para Group 3 y Group 4
weighted_sum_g3 = (data_in[common_genes_g3] * datatop_g3_indexed.loc[common_genes_g3, 'Mean_Abs_SHAP']).sum(axis=1)
weighted_sum_g4 = (data_in[common_genes_g4] * datatop_g4_indexed.loc[common_genes_g4, 'Mean_Abs_SHAP']).sum(axis=1)

total = weighted_sum_g3 + weighted_sum_g4

weighted_sum_g3_norm = weighted_sum_g3 / total
weighted_sum_g4_norm = weighted_sum_g4 / total

print("\nGroup 3 (norm):")
print(weighted_sum_g3_norm)

print("\nGroup 4 (norm):")
print(weighted_sum_g4_norm)

# ============================================================================
# VISUALIZACIONES UMAP
# ============================================================================

print("\n" + "="*80)
print("GENERANDO VISUALIZACIONES UMAP")
print("="*80 + "\n")

# Preparar datos para visualización
# Asegurarse de que data_clean tiene índice apropiado
if not data_clean.index.equals(labels_clean.index):
    data_clean.index = labels_clean.index

# 1. UMAP con clasificación discreta por grupos
print("1. Generando UMAP con clasificación por grupos...")
class_names_inv = {0: 'WNT', 1: 'SHH', 2: 'Group 3', 3: 'Group 4', 4: 'G3-G4'}
labels_str = labels_clean.map(class_names_inv)
labels_str.name = 'Subgroup'

colors_dict = {
    'WNT': '#1f77b4',      # azul
    'SHH': '#ff7f0e',      # naranja
    'Group 3': '#2ca02c',  # verde
    'Group 4': '#d62728',  # rojo
    'G3-G4': '#9467bd'     # púrpura
}

plot_umap_binary(
    data=data_clean,
    clinical=labels_str,
    colors_dict=colors_dict,
    n_components=2,
    save_fig=True,
    save_as="umap_subgroups",
    seed=42,
    title='UMAP: Medulloblastoma Subgroups',
    show=True,
    marker_size=10
)

# 2. UMAP con puntuación continua de Group 3
print("\n2. Generando UMAP con puntuación de Group 3...")
# Crear una Serie con las puntuaciones de Group 3
g3_scores = pd.Series(weighted_sum_g3_norm.values, index=data_in.index)
g3_scores.name = 'Group 3 Score'

plot_umap_spectrum(
    data=data_in,
    clinical=g3_scores,
    colormap='Reds',
    n_components=2,
    save_fig=True,
    save_as="umap_group3_score",
    seed=42,
    title='UMAP: Group 3 Signature Score',
    show=True,
    marker_size=10,
    color_range=[0, 1],
    colorbar_title='Group 3 Score'
)

# 3. UMAP con puntuación continua de Group 4
print("\n3. Generando UMAP con puntuación de Group 4...")
g4_scores = pd.Series(weighted_sum_g4_norm.values, index=data_in.index)
g4_scores.name = 'Group 4 Score'

plot_umap_spectrum(
    data=data_in,
    clinical=g4_scores,
    colormap='Blues',
    n_components=2,
    save_fig=True,
    save_as="umap_group4_score",
    seed=42,
    title='UMAP: Group 4 Signature Score',
    show=True,
    marker_size=10,
    color_range=[0, 1],
    colorbar_title='Group 4 Score'
)

# 4. UMAP 3D con clasificación por grupos
print("\n4. Generando UMAP 3D con clasificación por grupos...")
plot_umap_binary(
    data=data_clean,
    clinical=labels_str,
    colors_dict=colors_dict,
    n_components=3,
    save_fig=True,
    save_as="umap_3d_subgroups",
    seed=42,
    title='3D UMAP: Medulloblastoma Subgroups',
    show=True,
    marker_size=6
)

print("\n" + "="*80)
print("VISUALIZACIONES COMPLETADAS")
print("="*80)