# interface.py

import tkinter as tk
from tkinter import messagebox, Label, Button
from PIL import Image, ImageTk
from fpdf import FPDF
import joblib
import os
import pandas as pd

class Interfaz:
    def __init__(self, root):
        self.root = root
        self.root.title("Generador de Reporte de Emisiones")

        # Ruta de la imagen
        self.image_path = "C:\\Users\Daniel\Documents\APRENDIZAJE AUTOMATICO\V2\Imagen1.jpg"

        # Cargar imagen original
        self.image = Image.open(self.image_path)
        self.photo = ImageTk.PhotoImage(self.image)

        # Ajustar el tamaño de la ventana a la imagen original
        self.root.geometry(f"{self.image.width}x{self.image.height}")

        # Crear un Label para mostrar la imagen
        self.label = Label(self.root, image=self.photo)
        self.label.pack(fill=tk.BOTH, expand=tk.YES)

        # Botón para cerrar la ventana
        self.close_button = Button(self.root, text="Cerrar", command=self.root.quit)
        self.close_button.pack()

        # Permitir que la imagen se redimensione cuando la ventana cambia de tamaño
        self.root.bind("<Configure>", self.resize_image)

        # Etiqueta e ingreso del país
        self.label_pais = tk.Label(self.root, text="Ingrese el país:")
        self.label_pais.pack(pady=10)

        self.entry_pais = tk.Entry(self.root)
        self.entry_pais.pack(pady=10)

        # Botón para generar el reporte
        self.btn_generar = tk.Button(self.root, text="Generar Reporte", command=self.generar_reporte)
        self.btn_generar.pack(pady=10)

    def resize_image(self, event):
        # Obtener el tamaño actual de la ventana
        new_width = event.width
        new_height = event.height

        # Redimensionar la imagen
        resized_image = self.image.resize((new_width, new_height))
        self.photo = ImageTk.PhotoImage(resized_image)

        # Actualizar el Label con la nueva imagen escalada
        self.label.config(image=self.photo)
        self.label.image = self.photo  # Para evitar que la imagen sea recolectada por el Garbage Collector

    def generar_reporte(self):
        pais = self.entry_pais.get()
        try:
            modelo = self.load_model()  # Cargar el modelo entrenado
            datos = pd.DataFrame([[1.0, 2.0]], columns=['Savanna_fires', 'Forest_fires', 'Crop_Residues',
                                                        'Rice_Cultivation', 'Drained_organic_soils_(CO2)',
                                                        'Pesticides_Manufacturing', 'Food_Transport', 'Forestland',
                                                        'Net_Forest_conversion', 'Food_Household_Consumption',
                                                        'Food_Retail', 'Onfarm_Electricity_Use', 'Food_Packaging',
                                                        'Agrifood_Systems_Waste_Disposal', 'Food_Processing',
                                                        'Fertilizers_Manufacturing', 'IPPU', 'Manure_applied_to_Soils',
                                                        'Manure_left_on_Pasture', 'Manure_Management', 'Fires_in_organic_soils',
                                                        'Fires_in_humid_tropical_forests', 'Onfarm_energy_use',
                                                        'Rural_population', 'Urban_population', 'Total_Population__Male',
                                                        'Total_Population__Female', 'Average_Temperature'])  # Datos de entrada simulados
            self.generar_pdf(pais, modelo, datos)
        except FileNotFoundError as e:
            messagebox.showerror("Error", str(e))

    def load_model(self):
        if os.path.exists('model.pkl'):
            return joblib.load('model.pkl')
        else:
            raise FileNotFoundError("El archivo 'model.pkl' no existe. Entrena el modelo primero.")

    def generar_pdf(self, pais, modelo, datos):
        pdf = FPDF()
        pdf.add_page()

        # Título del PDF
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(200, 10, f'Reporte de Emisiones de {pais}', ln=True, align='C')

        # Generar predicción con el modelo
        predicciones = modelo.predict(datos)

        # Agregar recomendaciones y análisis (placeholder)
        pdf.set_font('Arial', '', 12)
        pdf.ln(10)
        pdf.multi_cell(0, 10, f'Las emisiones estimadas para {pais} son {predicciones[0]}...')

        # Guardar el PDF
        pdf.output(f"reporte_{pais}.pdf")
        messagebox.showinfo("Reporte generado", f"El reporte de {pais} ha sido creado.")


if __name__ == "__main__":
    root = tk.Tk()
    app = Interfaz(root)
    root.mainloop()
