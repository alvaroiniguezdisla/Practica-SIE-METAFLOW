from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar los datos
print("📥 Cargando datos de entrenamiento y test...")
train_df = pd.read_csv('data/train_preprocessed.csv')
test_df = pd.read_csv('data/test_preprocessed.csv')

X_train = train_df[['team', 'opponent']]
y_train = train_df['Resultado']

X_test = test_df[['team', 'opponent']]
y_test = test_df['Resultado']

# 2. Entrenar el modelo
print("🚀 Entrenando modelo RandomForestClassifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 3. Evaluación
y_pred = clf.predict(X_test)
print("📊 Reporte de evaluación del modelo:")
print(classification_report(y_test, y_pred, target_names=["Win", "Draw", "Loss"]))

# 4. Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Win", "Draw", "Loss"])
disp.plot(cmap='Blues')
plt.title("🔍 Matriz de Confusión")
plt.savefig("data/matriz_confusion.png")
plt.close()

# 5. Importancia de características
importancia = clf.feature_importances_
print("📌 Importancia de características:")
print(f"Team: {importancia[0]:.4f}")
print(f"Opponent: {importancia[1]:.4f}")

# 6. Guardar el modelo entrenado
joblib.dump(clf, 'data/modelo_entrenado.pkl')
print("✅ Modelo guardado en 'data/modelo_entrenado.pkl'")
