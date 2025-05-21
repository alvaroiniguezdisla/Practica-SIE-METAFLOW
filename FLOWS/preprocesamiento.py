from metaflow import FlowSpec, step
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class InitFlow(FlowSpec):

    @step
    def start(self):
        print("📥 Cargando CSV original...")
        self.data = pd.read_csv('data/premier_league.csv')
        self.data['date'] = pd.to_datetime(self.data['date'])  # asegurar fechas aunque ya no se use
        self.train, self.test = train_test_split(self.data, test_size=0.2, random_state=42)
        self.next(self.ingenieria_de_datos)

    @step
    def ingenieria_de_datos(self):
        print("🔨 Ingeniería de datos simplificada...")

        columnas_utiles = ['team', 'opponent', 'venue', 'result']
        self.train = self.train[columnas_utiles].dropna()
        self.test = self.test[columnas_utiles].dropna()

        # Guardar copia original
        self.train.to_csv('data/train_with_names.csv', index=False)
        self.test.to_csv('data/test_with_names.csv', index=False)

        # Codificar 'team' y 'opponent'
        equipos_combinados = pd.concat([
            self.train['team'], self.train['opponent'],
            self.test['team'], self.test['opponent']
        ])
        equipo_encoder = LabelEncoder().fit(equipos_combinados)

        self.train['team'] = equipo_encoder.transform(self.train['team'])
        self.train['opponent'] = equipo_encoder.transform(self.train['opponent'])
        self.test['team'] = equipo_encoder.transform(self.test['team'])
        self.test['opponent'] = equipo_encoder.transform(self.test['opponent'])

        # Codificar 'venue': Home = 0, Away = 1
        self.train['venue'] = self.train['venue'].map({'Home': 0, 'Away': 1})
        self.test['venue'] = self.test['venue'].map({'Home': 0, 'Away': 1})

        # Codificar resultado: W=0, D=1, L=2
        resultado_map = {'W': 0, 'D': 1, 'L': 2}
        self.train['Resultado'] = self.train['result'].map(resultado_map)
        self.test['Resultado'] = self.test['result'].map(resultado_map)

        # Eliminar 'result'
        self.train.drop(columns=['result'], inplace=True)
        self.test.drop(columns=['result'], inplace=True)

        # Guardar datasets finales
        self.train.to_csv('data/train_preprocessed.csv', index=False)
        self.test.to_csv('data/test_preprocessed.csv', index=False)

        print("✅ Ejemplo de datos procesados:")
        print(self.train.head())

        self.train_processed = self.train
        self.test_processed = self.test

        self.next(self.end)

    @step
    def end(self):
        print("🏁 ¡Preprocesamiento completado!")

if __name__ == '__main__':
    InitFlow()
