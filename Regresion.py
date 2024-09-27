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
plt.show() 
plt.close()





####################################################
y = df.pop('total_emission')
y = pd.DataFrame(y, columns = ['total_emission'])
df = pd.concat([df, y], axis=1)


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

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_val)



def prepare_and_predict_with_model(model, df, X_train, y_train):
    global_df = df.copy()
    last_year = global_df['Year'].max()

    
    global_df = global_df[['Urban_population', 'Onfarm_energy_use', 'IPPU', 'Manure_Management', 'Food_Processing', 'Average_Temperature', 'Year', 'Food_Household_Consumption', 'total_emission']]
    global_df = global_df.drop(columns=['total_emission'], errors='ignore')

    
    new_years = pd.DataFrame({'Year': np.arange(last_year + 1, last_year + 6)})
    extended_df = pd.concat([global_df, new_years], ignore_index=True)


    if not hasattr(model, "coef_"):
        model.fit(X_train, y_train)
    for col in X_train.columns:
        if col != 'Year' and col in extended_df.columns:
            extended_df[col].fillna(extended_df[col].mean(), inplace=True)


    new_years_df = extended_df[extended_df['Year'] > last_year]

    predictions = model.predict(new_years_df[X_train.columns])

 
    results = pd.DataFrame({
        'Year': new_years_df['Year'],
        'total_emission': predictions 
    })

    
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.lineplot(data=results, x='Year', y='total_emission', ax=ax, marker='o', label='Predictions')


    plt.title('Predicted Total Emissions for the Next 5 Years', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Predicted Total Emissions', fontsize=14)


    plt.grid(True)
    plt.xticks(results['Year'])
    plt.legend()
    plt.show()

    return results

next_5_years_results = prepare_and_predict_with_model(model, df, X_train, y_train)
