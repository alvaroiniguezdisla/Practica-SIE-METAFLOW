import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Cargar modelo entrenado
modelo = joblib.load('data/modelo_entrenado.pkl')

# Cargar dataset original solo para codificar nombres de equipos
df = pd.read_csv('data/premier_league.csv').dropna(subset=['team', 'opponent', 'venue'])
equipo_encoder = LabelEncoder().fit(pd.concat([df['team'], df['opponent']]))

# 🎮 Interfaz de usuario
print("\n🎮 SIMULADOR DE RESULTADOS - PREMIER LEAGUE")
print("============================================")

team_name = input("👤 Equipo principal (ej: Arsenal): ").strip()
opponent_name = input("👥 Rival (ej: Chelsea): ").strip()
local = input("📍 ¿Quién juega en casa? (1 = el equipo principal, 2 = el rival): ").strip()

# Validar y codificar equipos
try:
    team_encoded = equipo_encoder.transform([team_name])[0]
    opponent_encoded = equipo_encoder.transform([opponent_name])[0]
except ValueError:
    print("\n❌ Error: Uno de los equipos no está en el dataset.")
    exit(1)

# Codificar venue
if local == '1':
    venue_value = 0
elif local == '2':
    venue_value = 1
else:
    print("\n❌ Opción inválida. Debe ser 1 o 2.")
    exit(1)

# Crear entrada para el modelo
entrada = pd.DataFrame([{
    'team': team_encoded,
    'opponent': opponent_encoded,
    'venue': venue_value
}])

# Predecir resultado
prediccion = modelo.predict(entrada)[0]
mapa_resultado = {0: 'GANARÁ', 1: 'EMPATARÁ', 2: 'PERDERÁ'}

# Mostrar resultado
print("\n🔍 Simulando partido...\n")
print(f"📢 El equipo '{team_name}' probablemente **{mapa_resultado[prediccion]}** contra '{opponent_name}'\n")
