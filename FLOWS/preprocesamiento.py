from metaflow import FlowSpec, step
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class InitFlow(FlowSpec):
        @step
        def start(self):
            self.data = pd.read_csv('data/premier_league.csv')
            self.train, self.test = train_test_split(self.data, test_size=0.2,random_state=42)
            self.next(self.ingenieria_de_datos)


        @step
        def ingenieria_de_datos(self):
            print("🔨 Ingeniería de datos...")
           
            columnas_necesarias = ['team', 'opponent', 'result']
            # Filtrar las columnas necesarias para ambos conjuntos de datos directamente
            self.train = self.train[columnas_necesarias]
            self.test = self.test[columnas_necesarias]


            # Guardamos una copia con nombres originales (si es necesario para auditoría)
            self.train.to_csv('data/train_with_names.csv', index=False)
            self.test.to_csv('data/test_with_names.csv', index=False)


            # Codificamos los nombres de los equipos para el modelo
            equipos_combinados = pd.concat([self.train['team'], self.train['opponent'], self.test['team'], self.test['opponent']])
            equipo_encoder = LabelEncoder()
            equipo_encoder.fit(equipos_combinados)


            # Codificación de los equipos en los conjuntos de entrenamiento y test
            self.train['team'] = equipo_encoder.transform(self.train['team'])
            self.train['opponent'] = equipo_encoder.transform(self.train['opponent'])
            self.test['team'] = equipo_encoder.transform(self.test['team'])
            self.test['opponent'] = equipo_encoder.transform(self.test['opponent'])


            # Mapear los resultados
            resultado_map = {'W': 0, 'D': 1, 'L': 2}
            self.train['Resultado'] = self.train['result'].map(resultado_map)
            self.test['Resultado'] = self.test['result'].map(resultado_map)


            # Eliminar la columna 'result' ya que ya tenemos la columna 'Resultado'
            self.train = self.train.drop(columns=['result'])
            self.test = self.test.drop(columns=['result'])


            # Mostrar los primeros registros procesados para verificar
            print("✅ Datos preprocesados:")
            print(self.train.head())
            print(self.test.head())


            # Guardamos los archivos limpios y codificados para entrenamiento
            self.train.to_csv('data/train_preprocessed.csv', index=False)
            self.test.to_csv('data/test_preprocessed.csv', index=False)


            # Guardar los datos preprocesados en el flujo
            self.train_processed = self.train
            self.test_processed = self.test


            self.next(self.end)
           


        @step
        def end(self):
            print("✅ ¡Preprocesamiento completado!")


if __name__ == '__main__':
    InitFlow()

