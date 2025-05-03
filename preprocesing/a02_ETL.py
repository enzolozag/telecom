
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys, os
import numpy as np



# Configuración de rutas del proyecto
project_root = Path(os.getcwd()).parent
sys.path.append(str(project_root))

# Leer dataset procesado
df_encoded = pd.read_feather(project_root/'datasets/a01_ETL.feather')

# Importación de funciones personalizadas
from functions.f01_preprocesing_functions import df_explore, df_ValuesCheck


# ====================
# ANÁLISIS DE BALANCE DE CLASES
# ====================
print("\nAnalizando balance de clases (Churn):")

# Distribución de la variable target
churn_dist = df_encoded['Churn'].value_counts(normalize=True) * 100
print(f"\nDistribución de Churn:\n{churn_dist}")

# Visualización
plt.figure(figsize=(8, 5))
sns.countplot(x='Churn', data=df_encoded)
plt.title('Distribución de Clases (Churn)')
plt.xlabel('Churn (0: No, 1: Sí)')
plt.ylabel('Cantidad de clientes')
plt.show()

# Guardar el porcentaje de churn para referencia
churn_percentage = churn_dist[1]
print(f"\nPorcentaje de clientes que hicieron churn: {churn_percentage:.2f}%")

# ====================
# MATRIZ DE CORRELACIÓN
# ====================
print("\nCalculando matriz de correlación...")

# Calcular matriz de correlación
corr_matrix = df_encoded.corr()

# Visualizar matriz de correlación con Churn
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix[['Churn']].sort_values(by='Churn', ascending=False), 
            annot=True, cmap='coolwarm', center=0)
plt.title('Correlación de variables con Churn')
plt.show()

# Visualizar matriz completa (puede ser grande)
plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
            fmt='.2f', linewidths=0.5)
plt.title('Matriz de Correlación Completa')
plt.show()

# ====================
# FEATURE ENGINEERING ADICIONAL
# ====================
print("\nCreando características adicionales...")

# Ratio TotalCharges/MonthlyCharges
df_encoded['ChargeRatio'] = df_encoded['TotalCharges'] / df_encoded['MonthlyCharges']
df_encoded['ChargeRatio'] = df_encoded['ChargeRatio'].replace([np.inf, -np.inf], np.nan).fillna(0)

# Cantidad de servicios adicionales contratados
services = ['OnlineSecurity_bin', 'OnlineBackup_bin', 'DeviceProtection_bin',
            'TechSupport_bin', 'StreamingTV_bin', 'StreamingMovies_bin']
df_encoded['TotalServices'] = df_encoded[services].sum(axis=1)

# Cliente de larga duración (más de 24 meses)
df_encoded['LongTermCustomer'] = (df_encoded['PlanDurationMonths'] > 24).astype(int)

# Verificar nuevas características
print("\nNuevas características creadas:")
print("- ChargeRatio: Ratio TotalCharges/MonthlyCharges")
print("- TotalServices: Cantidad de servicios adicionales contratados")
print("- LongTermCustomer: Indicador de cliente de larga duración (>24 meses)")

# ====================
# NORMALIZACIÓN DE VARIABLES
# ====================
from sklearn.preprocessing import StandardScaler

print("\nNormalizando variables numéricas...")

# Identificar columnas numéricas (excluyendo variables binarias y Churn)
numeric_cols = ['MonthlyCharges', 'TotalCharges', 'PlanDurationMonths', 
                'ChargeRatio', 'TotalServices']

# Aplicar StandardScaler
scaler = StandardScaler()
df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])

# Verificar normalización
print("\nEstadísticas después de normalización:")
print(df_encoded[numeric_cols].describe().loc[['mean', 'std']])

# ====================
# GUARDADO FINAL
# ====================
print("\nGuardando dataset final con nuevas características...")

# Guardar versión con nuevas características
df_encoded.to_feather(f'{project_root}/datasets/dataframe_enriched.feather')

# Opcional: Guardar también una versión CSV
# df_encoded.to_csv(f'{project_root}/datasets/dataframe_enriched.csv', index=False)

print("""
Proceso completado exitosamente!
Dataset guardado con:
- Validación de datos
- Análisis de balance de clases
- Nuevas características de ingeniería
- Variables numéricas normalizadas
""")


# ====================
# ANÁLISIS ADICIONAL
# ====================
# Análisis de importancia de características preliminar
from sklearn.ensemble import RandomForestClassifier

# Preparar datos
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

# Entrenar modelo rápido para ver importancia de características
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Visualizar importancia de características
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
plt.title('Importancia de Características (Random Forest)')
plt.show()