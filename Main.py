import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter import Label
from PIL import Image, ImageTk
import os
import sys
from tkinter import filedialog
import matplotlib.pyplot as plt
import shutil
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os
from fpdf import FPDF
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import learning_curve


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

all_contet = []
all_images = []
img_idx = [0]
displayed_img = []
temp_results_dir = None
ppd_images = []
temp_csv_path = None
loop = None
global_df = None

### REGRESION ###
seed=42

df1=pd.read_csv('D:\\Docs_Sacave\\Desktop\\Semestre\\archive (4)\\Agrofood_co2_emission.csv',header=None)
df1.head().style.set_properties(**{'background-color': 'white',
                            'color': 'black',
                            'border-color': 'black'})

df_src=pd.read_csv(os.path.join('Agrofood_co2_emission.csv'))
df_src.info()
df_src['Area'].unique()
df_src['Area'].value_counts()
df_src.isnull().sum()
df_src.describe()

df_src["Savanna fires"].fillna(df_src["Savanna fires"].mean(), inplace = True)
df_src["Forest fires"].fillna(df_src["Forest fires"].mean(), inplace = True)
df_src["Crop Residues"].fillna(df_src["Crop Residues"].mean(), inplace = True)
df_src["Forestland"].fillna(df_src["Forestland"].mean(), inplace = True)
df_src["Net Forest conversion"].fillna(df_src["Net Forest conversion"].mean(), inplace = True)
df_src["Food Household Consumption"].fillna(df_src["Food Household Consumption"].mean(), inplace = True)
df_src["IPPU"].fillna(df_src["IPPU"].mean(), inplace = True)
df_src["Manure applied to Soils"].fillna(df_src["Manure applied to Soils"].mean(), inplace = True)
df_src["Manure Management"].fillna(df_src["Manure Management"].mean(), inplace = True)
df_src["Fires in humid tropical forests"].fillna(df_src["Fires in humid tropical forests"].mean(), inplace = True)
df_src["On-farm energy use"].fillna(df_src["On-farm energy use"].mean(), inplace = True)
df_src.columns = df_src.columns.str.replace(' ', '_')
df_src.columns = df_src.columns.str.replace('-', '')
df_src.columns = df_src.columns.str.replace(r'_\(CO2\)', '')
df_src.columns = df_src.columns.str.replace(r'_°C', '')

df_src = df_src.iloc[:,1:]

df_src.info()

df = df_src[['Urban_population', 'Onfarm_energy_use', 'IPPU', 'Manure_Management', 'Food_Processing', 'Average_Temperature', 'Year', 'Food_Household_Consumption', 'total_emission']]

sns.set_style("white")
fig, ax = plt.subplots(figsize=(16, 8))
sns.lineplot(data=df, x='Year', y='total_emission', ax=ax)
plt.title('Emisiones Totales por Año')
plt.xlabel('Año')
plt.ylabel('Emisiones Totales')
plt.grid()
plt.savefig('grafico_emisiones.png')  
plt.close()  # Cerrar la figura para liberar memoria
    ###########################################

sns.set_style("darkgrid")
fig, ax = plt.subplots(figsize=(16, 8))
if 'Average_Temperature' in df.columns:
    sns.lineplot(data=df, x='Year', y='Average_Temperature', ax=ax) 
    fig.suptitle('Temperatura medio a lo largo del tiempo')
else:
    print("La columna 'Average_Temperature' no existe en el DataFrame.")

image_path = 'grafico_temperatura.png'
plt.savefig(image_path)  
plt.close()

    #####################################################
plt.figure(figsize=(10, 6))
sns.barplot(x='Year', y='Food_Household_Consumption', data=df)
plt.title('Consumo de alimentos por hogar a lo largo de los años')
plt.xticks(rotation=45)
plt.tight_layout()

image_path = 'grafico_alimentos.png'
plt.savefig(image_path) 
plt.close()





    ####################################################
y = df.pop('total_emission')
y = pd.DataFrame(y, columns = ['total_emission'])
df = pd.concat([df, y], axis=1)


from sklearn.model_selection import train_test_split

global X_train, y_train

train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
print(f'Tamaño de entrenamiento: {len(train_df)}')
print(f'Tamaño de validación: {len(val_df)}')
print(f'Tamaño de prueba: {len(test_df)}')

X_train = train_df.drop('total_emission', axis=1)
y_train = train_df['total_emission']

X_test = val_df.drop('total_emission', axis=1)
y_test = val_df['total_emission']

X_val = test_df.drop('total_emission', axis=1)
y_val = test_df['total_emission']

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

def prepare_and_predict_with_model(model, X_train, y_train):
    global global_df
    dft = global_df  # Copiar para no modificar el DataFrame global directamente

        # Verificar los nombres de las columnas
    print("Columnas disponibles en dft:", dft.columns.tolist())

    dft.columns = dft.columns.str.replace(' ', '_')
    dft.columns = dft.columns.str.replace('-', '')
    dft.columns = dft.columns.str.replace(r'_\(CO2\)', '', regex=True)
    dft.columns = dft.columns.str.replace(r'_°C', '', regex=True)


        # Rellenar valores nulos
    dft["Food_Household_Consumption"].fillna(dft["Food_Household_Consumption"].mean(), inplace=True)
    dft["IPPU"].fillna(dft["IPPU"].mean(), inplace=True)
    dft["Manure_Management"].fillna(dft["Manure_Management"].mean(), inplace=True)
    dft["Onfarm_energy_use"].fillna(dft["Onfarm_energy_use"].mean(), inplace=True)

        # Eliminar la columna total_emission solo si existe
    dft = dft.drop(columns=['total_emission'], errors='ignore')

        # Verificar si la columna 'Year' está presente antes de intentar usarla
    if 'Year' not in dft.columns:
            print("La columna 'Year' no se encuentra en dft. Columnas disponibles:", dft.columns.tolist())
            return None  # O puedes lanzar una excepción, si lo prefieres

    last_year = dft['Year'].max()

    new_years = pd.DataFrame({'Year': np.arange(last_year + 1, last_year + 6)})
    extended_df = pd.concat([dft, new_years], ignore_index=True)

        # Comprobar si el modelo ya ha sido ajustado
    if not hasattr(model, "coef_"):
            model.fit(X_train, y_train)

    for col in X_train.columns:
        if col != 'Year' and col in extended_df.columns:
            extended_df[col].fillna(extended_df[col].mean(), inplace=True)

    new_years_df = extended_df[extended_df['Year'] > last_year]
    predictions = model.predict(new_years_df[X_train.columns])

    results = pd.DataFrame({
        'Year': new_years_df['Year'],'total_emission': predictions 
        })

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.lineplot(data=results, x='Year', y='total_emission', ax=ax, marker='o', label='Predictions')
    plt.title('Predicted Total Emissions for the Next 5 Years', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Predicted Total Emissions', fontsize=14)
    plt.grid(True)
    plt.xticks(results['Year'])
    plt.grid
    plt.savefig('Regression_5_years')

#place an image as the grid
def display_logo(url, row, column, num):
    img = Image.open(resource_path(url))
    img = img.resize((int(img.size[0]/num), int(img.size[1]/num)))
    img = ImageTk.PhotoImage(img)
    logo_label = Label(image=img, bg="")
    logo_label.image = img
    if num > 2.5:
        logo_label.grid(column=column, row = row)
    else:
        logo_label.grid(column=column, row=row)
    return logo_label

def display_icon(url, row, column, funct, Width, Height):
    icon = Image.open(resource_path(url))
    icon = icon.resize((Width, Height))
    icon = ImageTk.PhotoImage(icon)
    icon_label = Button(image=icon, command=funct, width=Width, height=Height, border="0")
    icon_label.image = icon
    icon_label.grid(column=column, row=row, padx=10, pady=10)
    return icon_label

def display_textbox1(content, ro, col, root):
    text_box = Text(root, height=20, width=20, padx=10, pady=10)
    text_box.insert(1.0, content)
    text_box.tag_configure("center", justify="center")
    text_box.tag_add("center", 1.0, "end")
    text_box.grid(column=col, row=ro, columnspan=1, rowspan=3, sticky="w", padx=10, pady=10)

def display_textbox(content, ro, col, root):
    text_box = Text(root, height=20, width=20, padx=10, pady=10)
    text_box.insert(1.0, content)
    text_box.tag_configure("center", justify="center")
    text_box.tag_add("center", 1.0, "end")
    text_box.grid(column=col, row=ro, sticky="nsew", padx=10, pady=10)

def display_label_for_tittle(content, ro, col, root):
   label = Label(root, text=content, font=("Arial", 24), bg=root.cget("bg"))  # bg=root.cget("bg") establece el fondo al color del root
   label.grid(row=ro, column=col)

def set_background_image(root, url):
    # Inicializamos la imagen
    bg_image = Image.open(resource_path(url))
    
    # Ajustamos la imagen al tamaño de la ventana
    bg_photo = ImageTk.PhotoImage(bg_image)
    
    # Creamos el Label con la imagen de fondo
    bg_label = tk.Label(root, image=bg_photo)
    bg_label.image = bg_photo
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)
    
    # Actualizar la imagen si cambia el tamaño de la ventana
    def resize_image(event):
        # Redimensionamos la imagen para que cubra el área de la ventana
        new_bg_image = bg_image.resize((event.width, event.height), Image.LANCZOS)
        new_bg_photo = ImageTk.PhotoImage(new_bg_image)
        bg_label.config(image=new_bg_photo)
        bg_label.image = new_bg_photo
    
    # Vincular el evento de redimensionamiento
    root.bind("<Configure>", resize_image)

def generar_pdf():
    class PDF(FPDF):
        # Footer personalizado
        def footer(self):
            # Posición: 1.5 cm desde abajo
            self.set_y(-20)
            # Fuente: Times itálica, tamaño 12
            self.set_font('Times', 'I', 12)
            # Número de página
            self.cell(w=0, h=10, txt='Página ' + str(self.page_no()) + '/{nb}', border=0, align='C')

    # Crear una instancia de la clase PDF
    pdf = PDF()

    # Configuración general del PDF
    pdf.alias_nb_pages()  # Activa el uso de nb_pages para el total de páginas
    pdf.add_page()

    # Agregar una imagen con un enlace (logo)
    url = 'https://github.com/KukiUrie/Fina-proyect-'
    pdf.image('Logo.png', x=5, y=5, w=40, h=40, link=url)  # Ajustamos la posición del logo

    # Título
    pdf.set_font('Times', 'B', 20)  # Negrita y tamaño 20
    pdf.set_xy(25, 20)  # Ajustamos la posición del título
    pdf.cell(w=0, h=10, txt='ANALISIS DE EMISIONES CO2', border=0, align='C')

    # Texto del análisis
    pdf.set_xy(10, 40)  # Ajustar la posición del texto después del título
    texto = '''¿Qué son las emisiones de CO2 y cómo afectan al medio ambiente?
El dióxido de carbono es un gas incoloro y denso que forma parte de la atmósfera terrestre, sin embargo, sus altas emisiones actuales son una real amenaza para el ambiente.

¿Cuál es el verdadero significado de CO2?
El carbono es uno de los elementos químicos, al igual que otros como el oxígeno o el nitrógeno. Se conoce como el cuarto elemento con mayor abundancia en el universo. Este elemento, al ser combinado con el oxígeno, se convierte en dióxido de carbono o CO2 y, al ser dos elementos abundantes en la Tierra, han conformado desde el inicio un equilibrio natural en la vida. Las plantas lo necesitan para hacer fotosíntesis y todos los seres vivos lo exhalamos al respirar.
'''

    texto2 = ''' 
Gráfico de emisiones totales por año: Las emisiones totales por año reflejan tendencias importantes que pueden indicar el impacto de políticas ambientales, avances tecnológicos y cambios en la actividad industrial y el consumo energético. Además, es crucial examinar los sectores que más contribuyen a estas emisiones.
Este análisis también permite contextualizar las fluctuaciones en las emisiones con eventos históricos como crisis económicas o pandemias, proporcionando una visión más completa del comportamiento ambiental a lo largo del tiempo.
'''

    texto3 = ''' 
Consumo de alimentos a lo largo de los años: muestra un aumento gradual, sugiere que el consumo ha ido creciendo de forma sostenida, posiblemente reflejando el crecimiento de la población o cambios en el acceso a los alimentos. Una tendencia descendente podría indicar una disminución en la demanda o problemas económicos. Picos o caídas abruptas señalarían eventos anómalos, como crisis alimentarias o políticas que impactaron el consumo. 
Este análisis, combinado con otros factores como el uso de energía agrícola o las emisiones de CO2, podría ayudar a entender mejor los patrones de consumo en contextos ambientales o económicos específicos.
'''

    texto4 = '''
Temperatura media: Este gráfico es una representación visual de la evolución de la temperatura media a lo largo del tiempo, mostrando el periodo desde aproximadamente 1990 hasta 2020. En el eje horizontal se encuentran los años, mientras que el eje vertical muestra la temperatura media. La línea azul indica cómo ha variado la temperatura, con fluctuaciones visibles, pero con una tendencia general de aumento a lo largo del tiempo, especialmente desde los años 2000. El área sombreada alrededor de la línea representa la incertidumbre o variabilidad en los datos, sugiriendo que la temperatura ha sido más consistente en los últimos años.
'''

    texto5 = '''
Recomendaciones para reducir las emisiones de CO2:
1. Transición a Energías Renovables: Sustituir fuentes de energía fósiles por energías renovables (solar, eólica, geotérmica) reduce significativamente las emisiones de CO2.
2. Eficiencia Energética: Implementar tecnologías y prácticas que optimicen el uso de energía en la industria, el transporte y los hogares puede reducir el consumo y las emisiones.
3. Movilidad Sostenible: Promover el uso de transporte público, bicicletas, y vehículos eléctricos. Implementar políticas para incentivar la reducción del uso de vehículos privados.
4. Agricultura Sostenible: Dado que el consumo de alimentos está vinculado al aumento de emisiones, se deben promover prácticas agrícolas que reduzcan el uso de fertilizantes y pesticidas químicos, así como el uso de métodos orgánicos y la reducción del desperdicio alimentario.
5. Forestar y Proteger Bosques: Los bosques son sumideros de carbono naturales. Invertir en reforestación y proteger áreas verdes existentes ayuda a absorber el CO2 de la atmósfera.
6. Economía Circular: Fomentar el reciclaje y la reutilización de materiales para disminuir la demanda de producción nueva, reduciendo así las emisiones asociadas a los procesos industriales.

¿Qué pasa si no se toman medidas?
De no controlarse, los efectos del cambio climático elevarán la temperatura media mundial por encima de los 3 °C y afectarán negativamente a todos los ecosistemas. Ya se puede observar cómo el cambio climático puede intensificar tormentas y catástrofes, así como hacer que amenazas como la escasez de alimentos y agua se conviertan en realidad y desemboquen en conflictos.

¿Hay alguna posible solución?
La solución al problema del cambio climático requiere una acción ambiciosa y coordinada a todos los niveles, ya que es un desafío global que afecta a todos los aspectos de la vida. Aunque se han observado avances significativos, como el rápido crecimiento de las inversiones en energías renovables y la adopción de nuevas tecnologías, aún queda un largo camino por recorrer. Un hito importante en la lucha contra el cambio climático fue la adopción del Acuerdo de París en diciembre de 2015, en el que los países firmantes asumieron el compromiso de tomar medidas concretas para abordar este problema. Sin embargo, para alcanzar los objetivos establecidos, es imprescindible intensificar los esfuerzos y la implementación de políticas más estrictas y eficaces.
Las empresas y los inversores juegan un papel crucial en esta transición. Deben asumir la responsabilidad de reducir sus emisiones, no solo como un acto de compromiso ético y social, sino también como una estrategia que resulta económicamente beneficiosa a largo plazo. Invertir en tecnologías limpias y sostenibles, y adoptar prácticas más respetuosas con el medio ambiente, no solo ayudará a proteger el planeta, sino que también ofrecerá ventajas competitivas en un mercado global que cada vez valora más la sostenibilidad.
'''

    texto6 = ''' El gráfico presenta las proyecciones de las emisiones totales para los próximos cinco años, con la intención de informar y crear conciencia sobre el crecimiento esperado de las emisiones, lo que es crucial para la planificación y formulación de políticas ambientales. La línea azul indica un aumento constante en las emisiones, sugiriendo que, si no se implementan cambios significativos, estas seguirán creciendo, lo que podría estar vinculado a factores como el crecimiento económico y la industrialización. Al proporcionar datos específicos para cada año, el gráfico permite a los responsables de la toma de decisiones evaluar el ritmo de este aumento y establecer metas claras para la reducción de emisiones, resaltando la importancia de la visualización de datos en la comunicación científica y la necesidad urgente de abordar el cambio climático.

     '''

    # Establecer fuente y agregar el primer texto
    pdf.set_font('Times', '', 12)  # Fuente normal y tamaño 12
    pdf.multi_cell(w=0, h=7, txt=texto, border=1, align='L')

    # Texto para introducir el gráfico de emisiones
    pdf.set_xy(10, 105)  # Ajustar la posición del texto del gráfico
    pdf.multi_cell(w=0, h=7, txt=texto2, border=0, align='L')

    # Agregar el gráfico de emisiones
    pdf.image('grafico_emisiones.png', x=15, y=170, w=180)  # Ajustamos la posición del gráfico debajo del texto

    # Agregar una nueva página para la siguiente sección
    pdf.add_page()

    # Texto para introducir el gráfico de temperatura
    pdf.set_xy(10, 10)  # Ajustar la posición del texto al inicio de la nueva página
    pdf.multi_cell(w=0, h=7, txt=texto4, border=0, align='L')

    # Agregar el gráfico de temperatura (en la nueva página)
    pdf.image('grafico_temperatura.png', x=5, y=80, w=200)  # Ajustamos la posición del gráfico de temperatura

    pdf.add_page()


    # Texto para introducir el gráfico de alimentos
    pdf.set_xy(10, 10)  # Ajustar la posición del texto al inicio de la nueva página
    pdf.multi_cell(w=0, h=7, txt=texto3, border=0, align='L')

    pdf.image('grafico_alimentos.png', x=5, y=90, w=200)  # Ajustamos la posición del gráfico de temperatura

    pdf.add_page()

    pdf.add_page

    # Texto para introducir el gráfico de prediccion
    pdf.set_xy(10, 10)  # Ajustar la posición del texto al inicio de la nueva página
    pdf.multi_cell(w=0, h=7, txt=texto6, border=0, align='L')

    # Agregar el gráfico de temperatura (en la nueva página)
    pdf.image('Regression_5_years.png', x=5, y=80, w=210)  # Ajustamos la posición del gráfico de temperatura


    pdf.add_page()

    pdf.set_font('Times', '', 12)  # Fuente normal y tamaño 12
    pdf.set_xy(10, 10)  # Ajustar la posición del texto al inicio de la nueva página
    pdf.multi_cell(w=0, h=7, txt=texto5, border=1, align='L')


    
    # Guardar el PDF en un archivo
    pdf.output('hoja.pdf', 'F')


root = Tk()
root.title("Proyect")
root.geometry('+%d+%d' % (10, 6))

set_background_image(root,'D:\\Docs_Sacave\\Desktop\\Semestre\\archive (4)\\greenhouse-gas-emissions.jpg.optimal.jpg')

root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_rowconfigure(2, weight=1)
root.grid_rowconfigure(3, weight=1)
root.grid_rowconfigure(4, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
root.grid_columnconfigure(2, weight=1)

#header = Frame(root, width=150, height=150, bg="lightblue")
#header.grid(columnspan=3, rowspan=2, row=0, sticky= "nsew")

tittle = display_label_for_tittle("Analysis of Co2 Emissions", 0, 1, root)
def cleanup_temp_dir():
    temp_dir = "temp"  # Carpeta de archivos temporales, por ejemplo
    if os.path.exists(temp_dir):
        # Eliminar la carpeta y su contenido
        shutil.rmtree(temp_dir)
        print(f"Directorio temporal '{temp_dir}' eliminado.")
    else:
        print(f"No se encontró el directorio '{temp_dir}'.")

def read_csv_file_btn():
    global global_df  # Usamos la variable global para guardar el DataFrame

    # Abrir cuadro de diálogo para seleccionar el archivo CSV
    file_path = filedialog.askopenfilename(
        title="Selecct an CSV file",
        filetypes=[("CSV files", "*.csv")]
    )
    
    if file_path:  # Si se selecciona un archivo
        try:
            global_df = pd.read_csv(file_path)  # Leer el CSV y almacenarlo en global_df
            messagebox.showinfo("Information", "File upload successfully")
        except Exception as e:
            messagebox.showerror("Error upload the file", str(e))
    else:
        messagebox.showwarning("No file selected", "Please, selecct an CSV file.")
    if global_df is None:
        status_label.config(text="No CSV file upload", fg="red")
    else:
        status_label.config(text="CSV file upload", fg="green")


def exit_application():
    root.after(0, cleanup_temp_dir)
    root.quit()
    root.destroy()

status_label = tk.Label(root, text="Import a CSV file")
status_label.grid(column=1, row=2, padx=10, pady=10)
read_btn = display_icon(resource_path('D:\\Docs_Sacave\\Desktop\\Semestre\\archive (4)\\Subir Dataset1.png'), 2, 0, lambda: read_csv_file_btn(), 150, 80)
process_btn = display_icon(resource_path('D:\\Docs_Sacave\\Desktop\\Semestre\\archive (4)\\Procesamiento de datos.png'), 3, 0, lambda: prepare_and_predict_with_model(model, X_train, y_train), 150, 80)
download_btn = display_icon(resource_path('D:\\Docs_Sacave\\Desktop\\Semestre\\archive (4)\\Descargar informe.png'), 4, 0, lambda: generar_pdf(), 150, 80)
exit_btn = display_icon(resource_path('D:\\Docs_Sacave\\Desktop\\Semestre\\archive (4)\\Salida.png'), 4, 1, lambda: exit_application(), 150, 80)
text1 = display_textbox1("From smog hanging over cities to smoke inside the home, air pollution poses a major threat to health across the globe.  Almost all of the global population (99%) are exposed to air pollution levels that put them at increased risk for diseases including heart disease, stroke, chronic obstructive pulmonary disease, cancer and pneumonia.  WHO monitors the exposure  levels and health impacts (i.e. deaths, DALYs) of air pollution at the national, regional and global level from ambient (outdoor) and household air pollution. Such estimates are used for official reporting like the world health statistics, and the Sustainable Development Goals.", 2, 4, root)
root.mainloop()
