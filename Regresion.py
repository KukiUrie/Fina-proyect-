import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

seed=42

df1=pd.read_csv('C:\\Users\Daniel\Documents\PROYECTO FINAL APRENDIZAJE AUTOMATICO\Agrofood_co2_emission.csv',header=None)
df1.head().style.set_properties(**{'background-color': 'white',
                           'color': 'black',
                           'border-color': 'black'})
#
df_src=pd.read_csv(os.path.join('Agrofood_co2_emission.csv'))
df_src.info()

######################################################################
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

#################################################################

df_src.columns = df_src.columns.str.replace(' ', '_')
df_src.columns = df_src.columns.str.replace('-', '')
df_src.columns = df_src.columns.str.replace(r'_\(CO2\)', '')
df_src.columns = df_src.columns.str.replace(r'_°C', '')

df_src = df_src.iloc[:,1:]

df_src.info()

##################################################################

variable_salida = 'total_emission'
caracteristicas = df_src.drop(columns=[variable_salida])

# Añade la variable de salida al DataFrame de características para la correlación
df_correlacion = pd.concat([caracteristicas, df_src[variable_salida]], axis=1)

# Paso 2: Calcula la correlación
correlacion = df_correlacion.corr()
correlacion_salida = correlacion[[variable_salida]].iloc[:-1, :]

# Paso 3: Crea el heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlacion_salida, annot=True, cmap='cividis', vmin=-1, vmax=1)
plt.title('Heatmap de Correlación entre total_emission y Características')
plt.show()


##############################################################################

variable_salida = 'total_emission'

# Selecciona las variables independientes
variables_independientes = df_src.drop(columns=[variable_salida])

# Configura el tamaño de la figura y el diseño de subgráficos
n = len(variables_independientes.columns)
ncols = 4  # Número de columnas en el diseño de subgráficos
nrows = int(np.ceil(n / ncols))  # Calcula el número de filas necesarias

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 5 * nrows))

# Aplana el array de ejes para facilitar la iteración
axes = axes.flatten()

# Itera sobre las variables independientes y crea gráficos de dispersión
for i, variable in enumerate(variables_independientes.columns):
    sns.scatterplot(x=df_src[variable], y=df_src[variable_salida], ax=axes[i])
    axes[i].set_title(f'{variable} vs {variable_salida}')
    axes[i].set_xlabel(variable)
    axes[i].set_ylabel(variable_salida)

# Desactiva los ejes restantes si hay menos variables que subgráficos
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

# Ajusta el layout para evitar superposiciones
plt.tight_layout()
plt.show()

sns.set_style("white")
fig, ax = plt.subplots(figsize=(16, 8))
sns.lineplot(data=df_src, x='Year', y='total_emission', ax=ax)

##########################################################################

y = df_src.pop('total_emission')
y = pd.DataFrame(y, columns = ['total_emission'])
df = pd.concat([df_src, y], axis=1)

#########################################################################

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


# Import statsmodels.formula.api
import statsmodels.formula.api as smf
OLSres = smf.ols('total_emission ~ Urban_population', data=train_df).fit()
print(OLSres.summary())

#######################################################################

from statsmodels.tools.eval_measures import rmse

# fit your model which you have already done

# now generate predictions
ypred = OLSres.predict(X_test)

# calc rmse
rmse_ols = rmse(y_test, ypred)
print(rmse_ols)

#####################################################################

OLS1 = smf.ols('total_emission ~ Urban_population', data=train_df).fit()
OLS2 = smf.ols('total_emission ~ Urban_population + Onfarm_energy_use + IPPU', data=train_df).fit()
OLS3 = smf.ols('total_emission ~ Urban_population + Onfarm_energy_use + IPPU + Manure_Management + Food_Processing', data=train_df).fit()

from statsmodels.tools.eval_measures import rmse

ypred = OLS2.predict(X_test)

# calc rmse
rmse_mlr2 = rmse(y_test, ypred)
print("OLS2 rmse", rmse_mlr2)
ypred = OLS3.predict(X_test)
rmse_mlr3 = rmse(y_test, ypred)
print("OLS3 rmse", rmse_mlr3)

#####################################################################

from statsmodels.iolib.summary2 import summary_col

info_dict={'BIC' : lambda x: f"{x.bic:.2f}", #dictionary. lambda is one way to define a function. include parameter and result. BIC in name of function. x is parameter. after : is function.
           # f"{x.bic:.2f} is result of formula
           #.2f means until second decimal place of float
           #plug in OLS1 for x, returns BIC of first regression. BIC similar to R^2
           'AIC' : lambda x: f"{x.aic:.2f}",
    'No. observations' : lambda x: f"{int(x.nobs):d}"}
        #nobs is number of observations

# "dictionary" is another way to store data, which use "keys" to index elements (instead of numbers): key-value pair

results_table = summary_col(results=[OLS1,OLS2,OLS3],
                            float_format='%0.2f', #specifies data/number type. f means float and include up to second decimal place
                            stars = True, #include stars
                            model_names=['Model 1', #names each model
                                         'Model 2',
                                         'Model 3'],
                            info_dict=info_dict,
                            regressor_order=['Intercept',
                                             'Top10perc',
                                             'Private',
                                             'Outstate',
                                             'Personal',
                                             'Expend'])

results_table.add_title('OLS Regressions') #adds title

print(results_table) #prints table
##############################################################################

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
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

###########################################################################

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

########################################################################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LinearRegression

# Asumiendo que ya tienes tus conjuntos de datos: X_train, y_train, X_val, y_val, X_test, y_test

# Inicializa el modelo
model = LinearRegression()

# Genera la curva de aprendizaje
train_sizes, train_scores, val_scores = learning_curve(
    model, X_train, y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,  # Utiliza validación cruzada
    scoring='r2',  # Métrica a evaluar
    n_jobs=-1  # Utiliza todos los núcleos disponibles
)

# Calcula la media y desviación estándar de las puntuaciones
train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Grafica la curva de aprendizaje
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Puntuación de Entrenamiento', color='blue')
plt.plot(train_sizes, val_mean, label='Puntuación de Validación', color='green')

# Grafica las bandas de incertidumbre
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.2)
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, color='green', alpha=0.2)

# Configuración del gráfico
plt.title('Curva de Aprendizaje')
plt.xlabel('Tamaño del Conjunto de Entrenamiento')
plt.ylabel('Puntuación R^2')
plt.legend()
plt.grid()
plt.show()


##################################################################

from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Define y ajusta el modelo Lasso
model = Lasso(alpha=0.5)  # Puedes ajustar el valor de alpha para regularización
model.fit(X_train, y_train)

# Realiza predicciones
y_pred = model.predict(X_test)

# Calcula métricas de rendimiento
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

# Crea un DataFrame con los resultados de las métricas
resultados = pd.DataFrame({
    'Métrica': ['MSE', 'RMSE', 'R^2'],
    'Valor': [mse, rmse, r2]
})

# Imprime los coeficientes del modelo
print("Coeficientes del modelo (Lasso):")
for name, coef in zip(X_train.columns, model.coef_):
    print(f'{name}: {coef}')

# Imprime el intercepto del modelo
print("\nIntercepto del modelo (Lasso):", model.intercept_)

# Imprime los resultados de la regresión
print("\nResultados de la regresión (Lasso):")
print(resultados)

##############################################################

from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Inicializa el modelo de regresión Lasso
model = Lasso(alpha=0.5)  # Puedes ajustar alpha según sea necesario

# Entrena el modelo con los datos de entrenamiento
model.fit(X_train, y_train)

# Realiza predicciones con los datos de validación
y_pred = model.predict(X_val)

# Calcula las métricas
mse = mean_squared_error(y_val, y_pred)
rmse = mean_squared_error(y_val, y_pred, squared=False)
r2 = r2_score(y_val, y_pred)

# Crea un DataFrame con los resultados
resultados = pd.DataFrame({
    'Métrica': ['MSE', 'RMSE', 'R^2'],
    'Valor': [mse, rmse, r2]
})

# Imprime los coeficientes del modelo
print("Coeficientes del modelo:")
for name, coef in zip(X_train.columns, model.coef_):
    print(f'{name}: {coef}')

# Imprime el intercepto del modelo
print("\nIntercepto del modelo:", model.intercept_)

# Imprime los resultados de la regresión
print("\nResultados de la regresión:")
print(resultados)

