from metaflow import FlowSpec, step
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from metaflow import FlowSpec, step
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class PreprocesamientoFlow(FlowSpec):

    @step
    def start(self):
        print("🔄 Leyendo los datos...")
        self.df = pd.read_csv('data/premier_league.csv')

        print("📋 Columnas originales:")
        print(self.df.columns)

        self.next(self.limpiar_datos)

    @step
    def limpiar_datos(self):
        print("🧹 Limpiando y codificando datos...")

        columnas_necesarias = ['team', 'opponent', 'result']
        self.df = self.df[columnas_necesarias]

        # Guardamos una copia con nombres originales
        self.df.to_csv('data/premier_league_con_nombres.csv', index=False)

        # Codificamos los nombres para el modelo
        equipos_combinados = pd.concat([self.df['team'], self.df['opponent']])
        equipo_encoder = LabelEncoder()
        equipo_encoder.fit(equipos_combinados)

        df_codificado = self.df.copy()
        df_codificado['team'] = equipo_encoder.transform(self.df['team'])
        df_codificado['opponent'] = equipo_encoder.transform(self.df['opponent'])

        resultado_map = {'W': 0, 'D': 1, 'L': 2}
        df_codificado['Resultado'] = df_codificado['result'].map(resultado_map)
        df_codificado = df_codificado.drop(columns=['result'])

        print("✅ Datos preprocesados:")
        print(df_codificado.head())

        # Guardamos el archivo limpio codificado para entrenamiento

        self.next(self.end)

    @step
    def end(self):
        print("✅ ¡Preprocesamiento completado!")

if __name__ == '__main__':
    PreprocesamientoFlow()
