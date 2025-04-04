from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd

# Cargamos los datos preprocesados
train_df = pd.read_csv('data/train_preprocessed.csv')
test_df = pd.read_csv('data/test_preprocessed.csv')

# Variables de entrada (X) y salida (y)
X_train = train_df[['team', 'opponent']]
y_train = train_df['Resultado']

X_test = test_df[['team', 'opponent']]
y_test = test_df['Resultado']

# Entrenamos el modelo
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Hacemos predicciones
y_pred = clf.predict(X_test)

# Mostramos evaluación
print("📊 Reporte de evaluación del modelo:")
print(classification_report(y_test, y_pred, target_names=["Win", "Draw", "Loss"]))
