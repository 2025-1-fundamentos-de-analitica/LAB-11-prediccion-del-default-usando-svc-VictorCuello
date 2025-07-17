# flake8: noqa: E501

import pandas as pd
import os
import json
import joblib
import zipfile

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Función para aplicar la limpieza de datos, replicando la lógica exitosa.
    """
    df_clean = df.copy()

    # Renombrar columna objetivo
    if "default payment next month" in df_clean.columns:
        df_clean.rename(columns={"default payment next month": "default"}, inplace=True)

    # Remover columna ID
    if "ID" in df_clean.columns:
        df_clean.drop(columns=["ID"], inplace=True)
        
    # Eliminar NaNs
    df_clean.dropna(inplace=True)

    # Eliminar registros con información no disponible en EDUCATION y MARRIAGE
    df_clean = df_clean[(df_clean["EDUCATION"] != 0) & (df_clean["MARRIAGE"] != 0)]

    # Agrupar categorías de EDUCATION > 4 en 4
    df_clean.loc[df_clean["EDUCATION"] > 4, "EDUCATION"] = 4
    
    return df_clean


def main():
    """
    Función principal que ejecuta todo el flujo de trabajo del modelo.
    """
    # --- Definir rutas y directorios 
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
        
    project_root = os.path.dirname(script_dir) 
    
    input_dir = os.path.join(project_root, "files", "input")
    model_dir = os.path.join(project_root, "files", "models")
    output_dir = os.path.join(project_root, "files", "output")
    
    model_path = os.path.join(model_dir, "model.pkl.gz")
    metrics_path = os.path.join(output_dir, "metrics.json")
    
    # Asegurarse de que los directorios de salida existan
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Paso 1: Cargar y limpiar los datasets desde los ZIP ---
    print("Paso 1: Limpiando los datos...")
    
    train_zip_path = os.path.join(input_dir, 'train_data.csv.zip')
    internal_train_filename = 'train_default_of_credit_card_clients.csv'
    
    with zipfile.ZipFile(train_zip_path, 'r') as z:
        with z.open(internal_train_filename) as f:
            df_train = pd.read_csv(f)

    test_zip_path = os.path.join(input_dir, 'test_data.csv.zip')
    internal_test_filename = 'test_default_of_credit_card_clients.csv'
    
    with zipfile.ZipFile(test_zip_path, 'r') as z:
        with z.open(internal_test_filename) as f:
            df_test = pd.read_csv(f)

    df_train_clean = clean_data(df_train)
    df_test_clean = clean_data(df_test)
    print("Datos limpios.")

    # --- Paso 2: Dividir los datasets en X e y ---
    print("Paso 2: Dividiendo los datos en X e y...")
    target_column = "default"
    
    x_train = df_train_clean.drop(columns=[target_column])
    y_train = df_train_clean[target_column]
    
    x_test = df_test_clean.drop(columns=[target_column])
    y_test = df_test_clean[target_column]
    print("División completada.")

    # --- Paso 3: Crear el pipeline 
    print("Paso 3: Creando el pipeline...")
    
    categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
    numeric_features = [col for col in x_train.columns if col not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('pca', PCA()),
        ('feature_selection', SelectKBest(score_func=f_classif)),
        ('svm', SVC(random_state=42))
    ])
    print("Pipeline creado.")
    
    # --- Paso 4: Optimizar hiperparámetros ---
    print("Paso 4: Entrenando el modelo con GridSearchCV...")
    
    param_grid = {
        'pca__n_components': [20, 25, 30],
        'feature_selection__k': [10, 15, 20],
        'svm__C': [1, 10],
        'svm__gamma': ['scale', 'auto']
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5, # Usamos 5 splits para que sea más rápido
        scoring='balanced_accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_
    print(f"Mejores parámetros encontrados: {grid_search.best_params_}")

    # --- Paso 5: Guardar el modelo ---
    print("Paso 5: Guardando el mejor modelo...")
    joblib.dump(best_model, model_path)
    print(f"Modelo guardado en: {model_path}")

    # --- Paso 6 y 7: Calcular y guardar métricas 
    print("Paso 6 y 7: Calculando y guardando métricas...")
    
    all_metrics_to_save = []
    
    for dataset_name, X, y in [('train', x_train, y_train), ('test', x_test, y_test)]:
        y_pred = best_model.predict(X)
        
        metrics_dict = {
            'dataset': dataset_name,
            'precision': precision_score(y, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1_score': f1_score(y, y_pred)
        }
        all_metrics_to_save.append(metrics_dict)
        
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        cm_dict = {
            'type': 'cm_matrix',
            'dataset': dataset_name,
            'true_0': {"predicted_0": int(tn), "predicte_1": int(fp)}, 
            'true_1': {"predicted_0": int(fn), "predicted_1": int(tp)}
        }
        all_metrics_to_save.append(cm_dict)

    # Guardado en formato JSON Lines (un objeto JSON por línea)
    with open(metrics_path, 'w') as f:
        for item in all_metrics_to_save:
            f.write(json.dumps(item) + '\n')
            
    print(f"Métricas guardadas en: {metrics_path}")
    print("¡Proceso completado!")


if __name__ == "__main__":
    main()