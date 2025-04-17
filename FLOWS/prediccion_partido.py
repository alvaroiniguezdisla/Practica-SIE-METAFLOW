import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Cargar modelo
modelo = joblib.load('data/modelo_entrenado.pkl')

# Cargar los nombres originales
nombres_df = pd.read_csv('data/train_with_names.csv')

# Crear diccionario nombre -> número
equipos = sorted(nombres_df['team'].unique())
equipo_a_num = {nombre: i for i, nombre in enumerate(equipos)}

# Input
local = input("Nombre del equipo local: ")
visitante = input("Nombre del equipo visitante: ")

# Validar
if local not in equipo_a_num or visitante not in equipo_a_num:
    print("❌ Uno o ambos equipos no están en el dataset.")
    print("Equipos disponibles:", list(equipo_a_num.keys()))
else:
    local_cod = equipo_a_num[local]
    visitante_cod = equipo_a_num[visitante]

    # Predecir
    pred = modelo.predict([[local_cod, visitante_cod]])[0]
    resultado = ["Gana el local", "Empate", "Gana el visitante"][pred]
    print("🔮 Resultado predicho:", resultado)
