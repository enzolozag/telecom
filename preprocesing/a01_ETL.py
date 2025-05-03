
"""
Script de preprocesamiento para el proyecto de pronóstico de cancelación de clientes de Interconnect

Objetivos:
1. Cargar y explorar los datos de diferentes fuentes
2. Realizar limpieza y transformación de datos
3. Enriquecer los datos con nuevas características
4. Preparar los datos para el modelado predictivo

Archivos de entrada:
- contract.csv: Información de contratos
- personal.csv: Datos personales de clientes
- internet.csv: Servicios de internet
- phone.csv: Servicios telefónicos

Archivo de salida:
- dataframe.feather: Dataset preprocesado listo para modelado
"""

# ====================
# IMPORTACIÓN DE LIBRERÍAS
# ====================
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys, os
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Configuración de rutas del proyecto
project_root = Path(os.getcwd()).parent
sys.path.append(str(project_root))

# Importación de funciones personalizadas
from functions.f01_preprocesing_functions import df_explore, df_ValuesCheck

# ====================
# CARGA DE DATOS
# ====================
print("Cargando datasets...")
contract = pd.read_csv(project_root/'datasets/contract.csv')
personal = pd.read_csv(project_root/'datasets/personal.csv')
internet = pd.read_csv(project_root/'datasets/internet.csv')
phone = pd.read_csv(project_root/'datasets/phone.csv')

# Exploración inicial de los datos
print("\nExploración inicial de los datasets:")
print("1. Dataset de contratos:")
df_explore(contract)
print("\n2. Dataset de datos personales:")
df_explore(personal)
print("\n3. Dataset de servicios de internet:")
df_explore(internet)
print("\n4. Dataset de servicios telefónicos:")
df_explore(phone)

# ====================
# PREPROCESAMIENTO DEL DATASET DE CONTRATOS
# ====================
print("\nPreprocesando dataset de contratos...")

# 1. Creación de variable target (Churn)
contract['Churn'] = np.where(contract['EndDate'] == 'No', 0, 1)

# 2. Conversión de tipos de datos
contract['EndDate'] = pd.to_datetime(contract['EndDate'], errors='coerce')
contract['BeginDate'] = pd.to_datetime(contract['BeginDate'], errors='coerce')
contract['TotalCharges'] = pd.to_numeric(contract['TotalCharges'], errors='coerce')

# 3. Cálculo de duración del plan en meses
LastDate = pd.to_datetime('2020-02-01')  # Fecha de referencia
contract['PlanDurationMonths'] = ((contract['EndDate']-contract['BeginDate']).dt.days)/30
contract.loc[contract['EndDate'].isna(), 'PlanDurationMonths'] = ((LastDate - contract['BeginDate']).dt.days)/30
contract['PlanDurationMonths'] = np.round(contract['PlanDurationMonths']).astype('Int64')

# 4. Imputación de valores faltantes en TotalCharges
mask = contract['TotalCharges'].isna()
contract.loc[mask, 'TotalCharges'] = contract.loc[mask, 'MonthlyCharges'] * contract.loc[mask, 'PlanDurationMonths']

# 5. Eliminación de columnas no relevantes
contract = contract.drop(['PaperlessBilling', 'PaymentMethod'], axis=1)

# Visualizaciones para entender los datos
print("\nVisualizando relaciones clave...")
plt.figure(figsize=(12, 5))

# Relación entre Churn y MonthlyCharges
plt.subplot(1, 2, 1)
sns.barplot(x='Churn', y='MonthlyCharges', data=contract)
plt.title('Cargo mensual promedio por estado de cliente')
plt.xlabel('Cliente activo (0) vs Churn (1)')
plt.ylabel('Cargo mensual promedio')

# Distribución de tipos de contrato
plt.subplot(1, 2, 2)
sns.countplot(x='Type', hue='Churn', data=contract)
plt.title('Distribución de tipos de contrato')
plt.xlabel('Tipo de contrato')
plt.ylabel('Número de clientes')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# ====================
# PREPROCESAMIENTO DE LOS DEMÁS DATASETS
# ====================
print("\nRevisando valores ausentes y duplicados")
# df_ValuesCheck(personal)
# df_ValuesCheck(internet)
# df_ValuesCheck(phone)

# ====================
# INTEGRACIÓN DE DATASETS
# ====================
print("\nIntegrando todos los datasets...")
df = contract.merge(personal, on='customerID', how='left')
df = df.merge(internet, on='customerID', how='left')
df = df.merge(phone, on='customerID', how='left')

# ====================
# LIMPIEZA FINAL
# ====================
# 1. Imputación de valores faltantes en servicios
columns_to_fill = [
    'MultipleLines', 'OnlineSecurity', 'DeviceProtection', 
    'OnlineBackup', 'TechSupport', 'StreamingTV', 'StreamingMovies'
]
df[columns_to_fill] = df[columns_to_fill].fillna('No')

# 2. Eliminación de columnas no relevantes
df = df.drop(['customerID', 'BeginDate', 'EndDate', 'gender'], axis=1)

# ====================
# ENCODING DE VARIABLES CATEGÓRICAS
# ====================
print("\nRealizando encoding de variables categóricas...")
df_encoded = df.copy()

# 1. One-Hot Encoding para variables nominales
nominal_cols = ['Partner', 'Dependents', 'InternetService', 'MultipleLines']
ohe = OneHotEncoder(drop='first', sparse_output=False)
ohe_result = ohe.fit_transform(df_encoded[nominal_cols])

# Nombres para las nuevas columnas
ohe_columns = []
for i, col in enumerate(nominal_cols):
    for cat in ohe.categories_[i][1:]:
        ohe_columns.append(f"{col}_{cat}")

# Creación de DataFrame con las nuevas columnas
ohe_df = pd.DataFrame(ohe_result, columns=ohe_columns, index=df_encoded.index)
df_encoded = pd.concat([df_encoded, ohe_df], axis=1)
df_encoded.drop(nominal_cols, axis=1, inplace=True)

# 2. Ordinal Encoding para variables ordinales
ordinal_cols = {'Type': ['Month-to-month', 'One year', 'Two year']}
ordinal_encoder = OrdinalEncoder(categories=[ordinal_cols['Type']])
df_encoded['Type_encoded'] = ordinal_encoder.fit_transform(df_encoded[['Type']])
df_encoded.drop('Type', axis=1, inplace=True)

# 3. Binary Encoding para servicios con respuestas Yes/No
binary_cols = [
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies'
]

for col in binary_cols:
    df_encoded[f"{col}_bin"] = df_encoded[col].map({'Yes': 1, 'No': 0})
    
df_encoded.drop(binary_cols, axis=1, inplace=True)

# ====================
# GUARDADO DEL DATASET FINAL
# ====================
print("\nGuardando dataset preprocesado...")
df_encoded.to_feather(f'{project_root}/datasets/a01_ETL.feather')
print("Proceso completado exitosamente!")