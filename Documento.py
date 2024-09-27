from fpdf import FPDF

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
    pdf.image('grafico_prediccion.png', x=5, y=80, w=210)  # Ajustamos la posición del gráfico de temperatura


    pdf.add_page()

    pdf.set_font('Times', '', 12)  # Fuente normal y tamaño 12
    pdf.set_xy(10, 10)  # Ajustar la posición del texto al inicio de la nueva página
    pdf.multi_cell(w=0, h=7, txt=texto5, border=1, align='L')


    
    # Guardar el PDF en un archivo
    pdf.output('hoja.pdf', 'F')

if __name__ == "__main__":
    generar_pdf()
