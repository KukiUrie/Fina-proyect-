import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import learning_curve

seed=42

df1=pd.read_csv('C:\\Users\Daniel\Documents\APRENDIZAJE AUTOMATICO\V2\Agrofood_co2_emission.csv',header=None)
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

variable_salida = 'total_emission'
caracteristicas = df_src.drop(columns=[variable_salida])

df_correlacion = pd.concat([caracteristicas, df_src[variable_salida]], axis=1)

correlacion = df_correlacion.corr()
correlacion_salida = correlacion[[variable_salida]].iloc[:-1, :]

plt.figure(figsize=(8, 6))
sns.heatmap(correlacion_salida, annot=True, cmap='cividis', vmin=-1, vmax=1)
plt.title('Heatmap de Correlación entre total_emission y Características')
plt.show()

variable_salida = 'total_emission'
variables_independientes = df_src.drop(columns=[variable_salida])


n = len(variables_independientes.columns)
ncols = 4  
nrows = int(np.ceil(n / ncols))  

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 5 * nrows))

axes = axes.flatten()

for i, variable in enumerate(variables_independientes.columns):
    sns.scatterplot(x=df_src[variable], y=df_src[variable_salida], ax=axes[i])
    axes[i].set_title(f'{variable} vs {variable_salida}')
    axes[i].set_xlabel(variable)
    axes[i].set_ylabel(variable_salida)

for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()
################################################
sns.set_style("white")
fig, ax = plt.subplots(figsize=(16, 8))
sns.lineplot(data=df_src, x='Year', y='total_emission', ax=ax)
plt.title('Emisiones Totales por Año')
plt.xlabel('Año')
plt.ylabel('Emisiones Totales')
plt.grid()
# Guardar el gráfico como imagen
plt.savefig('grafico_emisiones.png')  # Guarda el gráfico como 'grafico_emisiones.png'
plt.close()  # Cerrar la figura para liberar memoria
###########################################

sns.set_style("darkgrid")
fig, ax = plt.subplots(figsize=(16, 8))
if 'Average_Temperature' in df_src.columns:
    sns.lineplot(data=df_src, x='Year', y='Average_Temperature', ax=ax) 
    fig.suptitle('Temperatura medio a lo largo del tiempo')
else:
    print("La columna 'Average_Temperature' no existe en el DataFrame.")

image_path = 'grafico_temperatura.png'
plt.savefig(image_path)  
plt.close()

#####################################################
plt.figure(figsize=(10, 6))
sns.barplot(x='Year', y='Food_Household_Consumption', data=df_src)
plt.title('Consumo de alimentos por hogar a lo largo de los años')
plt.xticks(rotation=45)  # Rotar las etiquetas del eje X para acomodarlas mejor
plt.tight_layout()  # Asegurar que todo se vea bien ajustado

image_path = 'grafico_alimentos.png'
plt.savefig(image_path)  # Guardar la imagen primero
plt.show()  # Mostrar después de guardarla
plt.close()  # Luego cerrar la figura





####################################################
y = df_src.pop('total_emission')
y = pd.DataFrame(y, columns = ['total_emission'])
df = pd.concat([df_src, y], axis=1)


from sklearn.model_selection import train_test_split
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

X_train.head()

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
resultados = pd.DataFrame({
    'Métrica': ['MSE', 'RMSE', 'R^2'],
    'Valor': [mse, rmse, r2]
})

print("Coeficientes del modelo:")
for name, coef in zip(X_train.columns, model.coef_):
    print(f'{name}: {coef}')

print("\nIntercepto del modelo:", model.intercept_)
print("\nResultados de la regresión:")
print(resultados)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
rmse = mean_squared_error(y_val, y_pred, squared=False)
r2 = r2_score(y_val, y_pred)
resultados = pd.DataFrame({
    'Métrica': ['MSE', 'RMSE', 'R^2'],
    'Valor': [mse, rmse, r2]
})

print("Coeficientes del modelo:")
for name, coef in zip(X_train.columns, model.coef_):
    print(f'{name}: {coef}')

print("\nIntercepto del modelo:", model.intercept_)
print("\nResultados de la regresión:")
print(resultados)


model = LinearRegression()

train_sizes, train_scores, val_scores = learning_curve(
    model, X_train, y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,  
    scoring='r2',  
    n_jobs=-1  
)

train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Puntuación de Entrenamiento', color='blue')
plt.plot(train_sizes, val_mean, label='Puntuación de Validación', color='green')


plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.2)
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, color='green', alpha=0.2)

plt.title('Curva de Aprendizaje')
plt.xlabel('Tamaño del Conjunto de Entrenamiento')
plt.ylabel('Puntuación R^2')
plt.legend()
plt.grid()
plt.show()
