from metaflow import FlowSpec, step 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

class EntrenamientoFlow(FlowSpec):

    @step
    def start(self):
        print("📥 Cargando datasets preprocesados...")
        self.train = pd.read_csv('data/train_preprocessed.csv')
        self.test = pd.read_csv('data/test_preprocessed.csv')
        self.next(self.entrenar)

    @step
    def entrenar(self):
        print("🧠 Entrenando modelo...")

        features = ['team', 'opponent', 'venue']
        X_train = self.train[features]
        y_train = self.train['Resultado']
        X_test = self.test[features]
        y_test = self.test['Resultado']

        self.modelo = RandomForestClassifier(random_state=42)
        self.modelo.fit(X_train, y_train)

        joblib.dump(self.modelo, 'data/modelo_entrenado.pkl')
        print("✅ Modelo guardado en 'data/modelo_entrenado.pkl'")

        self.predicciones = self.modelo.predict(X_test)
        self.y_test = y_test

        self.next(self.evaluar)

    @step
    def evaluar(self):
        precision = accuracy_score(self.y_test, self.predicciones)
        print(f"📊 Precisión del modelo: {precision:.2f}")
        self.next(self.end)

    @step
    def end(self):
        print("🏁 ¡Entrenamiento completado!")

if __name__ == '__main__':
    EntrenamientoFlow()
