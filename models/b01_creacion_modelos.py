import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, average_precision_score,
                            classification_report, confusion_matrix)
import joblib
from pathlib import Path
import sys, os

# Modelos a evaluar
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Configuración
RANDOM_STATE = 42

# ====================
# CARGA DE DATOS PREPROCESADOS
# ====================

# Configuración de rutas del proyecto
project_root = Path(os.getcwd()).parent
sys.path.append(str(project_root))

def load_preprocessed_data():
    """Carga los conjuntos ya preparados"""
    try:
        features_train = pd.read_feather(project_root/'datasets/features_train.feather')
        features_valid = pd.read_feather(project_root/'datasets/features_valid.feather')
        target_train = pd.read_feather(project_root/'datasets/target_train.feather')['Churn']
        target_valid = pd.read_feather(project_root/'datasets/target_valid.feather')['Churn']
        
        print("Datos cargados correctamente:")
        print(f"Entrenamiento: {features_train.shape}, {target_train.shape}")
        print(f"Validación: {features_valid.shape}, {target_valid.shape}")
        
        # Calcular ratio para balanceo en XGBoost
        churn_ratio = (target_train == 0).sum() / (target_train == 1).sum()
        
        return features_train, features_valid, target_train, target_valid, churn_ratio
    except Exception as e:
        print("Error al cargar los datos:", e)
        raise

# ====================
# DEFINICIÓN DE MODELOS (SIN NORMALIZACIÓN)
# ====================
def get_models(churn_ratio):
    """
    Define los modelos a evaluar
    (Sin pipelines de normalización ya que los datos están pre-normalizados)
    """
    models = {
        'Regresión Logística': LogisticRegression(
            class_weight='balanced',
            random_state=RANDOM_STATE,
            max_iter=1000
        ),
        
        'Random Forest': RandomForestClassifier(
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_estimators=200
        ),
        
        'XGBoost': XGBClassifier(
            scale_pos_weight=churn_ratio,
            eval_metric='aucpr',
            use_label_encoder=False,
            random_state=RANDOM_STATE
        ),
        
        'SVM': SVC(
            class_weight='balanced',
            probability=True,
            random_state=RANDOM_STATE,
            kernel='rbf'
        ),
        
        'Red Neuronal': MLPClassifier(
            hidden_layer_sizes=(50,),
            early_stopping=True,
            random_state=RANDOM_STATE
        )
    }
    return models

# ====================
# EVALUACIÓN DE MODELOS
# ====================
def evaluate_model(model, features_train, features_valid, target_train, target_valid):
    """Evalúa un modelo y retorna métricas"""
    model.fit(features_train, target_train)
    target_pred = model.predict(features_valid)
    target_pred_proba = model.predict_proba(features_valid)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(features_valid)
    
    metrics = {
        'accuracy': accuracy_score(target_valid, target_pred),
        'precision': precision_score(target_valid, target_pred),
        'recall': recall_score(target_valid, target_pred),
        'f1': f1_score(target_valid, target_pred),
        'roc_auc': roc_auc_score(target_valid, target_pred_proba),
        'pr_auc': average_precision_score(target_valid, target_pred_proba)
    }
    
    return metrics, target_pred, target_pred_proba

# ====================
# VISUALIZACIÓN DE RESULTADOS
# ====================
def plot_metrics_comparison(metrics_df):
    """Visualiza comparación de métricas entre modelos"""
    plt.figure(figsize=(12, 6))
    metrics_df.drop(columns=['roc_auc', 'pr_auc']).plot(kind='bar', colormap='viridis')
    plt.title('Comparación de Modelos (Métricas Principales)')
    plt.ylabel('Puntaje')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# ====================
# FUNCIÓN PRINCIPAL
# ====================
def main():
    # Carga los datos con tus nombres específicos
    features_train, features_valid, target_train, target_valid, churn_ratio = load_preprocessed_data()
    
    # Obtener modelos (sin normalización)
    models = get_models(churn_ratio)
    
    # Evaluación de modelos
    results = []
    for model_name, model in models.items():
        print(f"\n=== Evaluando {model_name} ===")
        metrics, target_pred, _ = evaluate_model(
            model, 
            features_train, 
            features_valid, 
            target_train, 
            target_valid
        )
        results.append({'Modelo': model_name, **metrics})
        
        # Reporte de clasificación
        print(f"\nReporte de Clasificación - {model_name}:")
        print(classification_report(target_valid, target_pred))
        
        # Matriz de confusión
        plt.figure(figsize=(5, 5))
        sns.heatmap(
            confusion_matrix(target_valid, target_pred), 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            cbar=False
        )
        plt.title(f'Matriz de Confusión\n{model_name}')
        plt.xlabel('Predicho')
        plt.ylabel('Real')
        plt.show()
    
    # Resultados comparativos
    metrics_df = pd.DataFrame(results).set_index('Modelo')
    print("\n=== Resumen Comparativo ===")
    print(metrics_df.sort_values('f1', ascending=False))
    
    # Visualización
    plot_metrics_comparison(metrics_df)
    
    # Guardar el mejor modelo basado en F1-score
    best_model_name = metrics_df['f1'].idxmax()
    best_model = models[best_model_name]
    best_model.fit(features_train, target_train)  # Reentrenar con todos los datos
    joblib.dump(best_model, 'mejor_modelo_churn.pkl')
    print(f"\nModelo {best_model_name} guardado como 'mejor_modelo_churn.pkl'")

if __name__ == "__main__":
    main()