# Función para leer información de un dataframe ------------------


def df_explore(df):
    print("Información del DataFrame:")
    df.info()
    print()
    print("Primeras filas del DataFrame:")
    print(df.head())
    print()
    print("Descripción estadística:")
    print(df.describe(include="all"))


def df_error_checking(df,column='customerID'):
    print('Columnas con valores ausentes:',df.isna().sum())
    print()
    print('Filas totalmente duplicadas:',df.duplicated().sum())
    print()
    print(f'Duplicados en columna {column}:',df[f'{column}'].duplicated().sum())