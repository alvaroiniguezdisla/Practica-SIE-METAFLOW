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
        df = self.df[columnas_necesarias].copy()

        # Codificamos los nombres de los equipos
        equipos_combinados = pd.concat([df['team'], df['opponent']])
        equipo_encoder = LabelEncoder()
        equipo_encoder.fit(equipos_combinados)

        # Creamos columnas nuevas con los códigos numéricos
        df['team_num'] = equipo_encoder.transform(df['team'])
        df['opponent_num'] = equipo_encoder.transform(df['opponent'])

        # Convertimos 'result' (W/D/L) en números:
        resultado_map = {'W': 0, 'D': 1, 'L': 2}
        df['Resultado'] = df['result'].map(resultado_map)

        # Renombramos columnas para claridad
        df = df.rename(columns={
            'team': 'team_nombre',
            'opponent': 'opponent_nombre'
        })

        # Reordenamos columnas
        df = df[['team_nombre', 'team_num', 'opponent_nombre', 'opponent_num', 'Resultado']]

        print("✅ Datos preprocesados:")
        print(df.head())

        # Guardamos el dataset final con nombres + códigos
        df.to_csv('data/premier_league_limpio.csv', index=False)

        self.next(self.end)

    @step
    def end(self):
        print("✅ ¡Preprocesamiento completado!")

if __name__ == '__main__':
    PreprocesamientoFlow()
