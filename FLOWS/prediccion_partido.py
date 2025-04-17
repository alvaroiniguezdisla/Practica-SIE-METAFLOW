from metaflow import FlowSpec, step, Parameter
import pandas as pd

class PrediccionHistorialFlow(FlowSpec):
    equipo1 = Parameter('equipo1', help="Nombre del equipo local")
    equipo2 = Parameter('equipo2', help="Nombre del equipo visitante")

    @step
    def start(self):
        print("📥 Cargando dataset original...")
        self.df = pd.read_csv('data/premier_league.csv')
        self.next(self.predecir)

    @step
    def predecir(self):
        print(f"🔍 Buscando historial entre {self.equipo1} y {self.equipo2}...")

        df = self.df
        df['team'] = df['team'].str.strip()
        df['opponent'] = df['opponent'].str.strip()

        partidos = df[((df['team'] == self.equipo1) & (df['opponent'] == self.equipo2)) |
                      ((df['team'] == self.equipo2) & (df['opponent'] == self.equipo1))]

        if partidos.empty:
            print("❌ No hay historial entre esos equipos.")
            self.prediccion = "Sin datos suficientes"
        else:
            victorias = empates = derrotas = 0

            for _, row in partidos.iterrows():
                if row['team'] == self.equipo1:
                    if row['result'] == 'W': victorias += 1
                    elif row['result'] == 'D': empates += 1
                    else: derrotas += 1
                else:
                    if row['result'] == 'L': victorias += 1
                    elif row['result'] == 'D': empates += 1
                    else: derrotas += 1

            print(f"\n📊 Historial de {self.equipo1} vs {self.equipo2}:")
            print(f"✅ Victorias: {victorias}")
            print(f"➖ Empates: {empates}")
            print(f"❌ Derrotas: {derrotas}")

            if victorias > max(empates, derrotas):
                self.prediccion = f"{self.equipo1} probablemente GANARÁ"
            elif empates > max(victorias, derrotas):
                self.prediccion = "Probable EMPATE"
            else:
                self.prediccion = f"{self.equipo1} probablemente PERDERÁ"

        self.next(self.end)

    @step
    def end(self):
        print(f"\n📢 Predicción final: {self.prediccion}")

if __name__ == '__main__':
    PrediccionHistorialFlow()

