import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Cargar datos con nombres y datos preprocesados
nombres_df = pd.read_csv('data/train_with_names.csv')
train_df = pd.read_csv('data/train_preprocessed.csv')

# Crear diccionario nombre -> número
equipos = sorted(nombres_df['team'].unique())
equipo_a_num = {nombre: i for i, nombre in enumerate(equipos)}

# Pedir equipos
local = input("Nombre del equipo local: ")
visitante = input("Nombre del equipo visitante: ")

# Validar
if local not in equipo_a_num or visitante not in equipo_a_num:
    print("❌ Uno o ambos equipos no están en el dataset.")
    print("Equipos disponibles:", list(equipo_a_num.keys()))
else:
    # Codificar
    local_cod = equipo_a_num[local]
    visitante_cod = equipo_a_num[visitante]

    # Entrenar modelo
    X_train = train_df[['team', 'opponent']]
    y_train = train_df['Resultado']
    modelo = RandomForestClassifier(n_estimators=100)
    modelo.fit(X_train, y_train)

    # Hacer predicción
    pred = modelo.predict([[local_cod, visitante_cod]])[0]
    resultado_texto = ['Gana el local', 'Empate', 'Gana el visitante'][pred]
    print("🔮 Resultado predicho:", resultado_texto)
