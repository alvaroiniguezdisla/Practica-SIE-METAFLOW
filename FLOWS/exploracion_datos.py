import pandas as pd

# Cargar los datos
df = pd.read_csv('data/train_with_names.csv')

# Mostrar primeras filas
print("Primeras filas del dataset:")
print(df.head())

# Mostrar información general
print("\nInformación del dataset:")
print(df.info())

# Mostrar valores únicos de los equipos
print("\nEquipos únicos:")
print(df['team'].unique())
