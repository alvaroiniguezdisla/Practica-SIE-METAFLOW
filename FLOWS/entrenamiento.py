from metaflow import FlowSpec, step
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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

        X_train = self.train[['team', 'opponent']]
        y_train = self.train['Resultado']
        X_test = self.test[['team', 'opponent']]
        y_test = self.test['Resultado']

        modelo = RandomForestClassifier(random_state=42)
        modelo.fit(X_train, y_train)

        self.predicciones = modelo.predict(X_test)
        self.y_test = y_test
        self.modelo = modelo

        self.next(self.evaluar)

    @step
    def evaluar(self):
        print("📊 Evaluando precisión del modelo...")

        from sklearn.metrics import accuracy_score
        precision = accuracy_score(self.y_test, self.predicciones)

        print(f"✅ Precisión del modelo: {precision:.2f}")
        self.next(self.end)

    @step
    def end(self):
        print("🏁 ¡Entrenamiento completado!")

if __name__ == '__main__':
    EntrenamientoFlow()
