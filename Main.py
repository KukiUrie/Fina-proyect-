# main.py

import Regresion
import Interfaz

def main():
    # Si necesitas entrenar el modelo, descomenta las siguientes líneas:
    csv_path = r'C:\Users\Daniel\Documents\APRENDIZAJE AUTOMATICO\V2\Agrofood_co2_emission.csv'
    Regresion.train_model(csv_path)
    # Ejecutar la interfaz gráfica
    Interfaz.interfaz()

if __name__ == "__main__":
    main()


import pandas as pd

# Cargar el archivo CSV
data = pd.read_csv('C:\\Users\\Daniel\\Documents\\APRENDIZAJE AUTOMATICO\\V2\\Agrofood_co2_emission.csv')

# Imprimir los nombres de las columnas
print(data.columns)
