import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# def predecir(model_path, input_data_path):
#     # Cargar el modelo entrenado
#     model = joblib.load(model_path)

#     # Cargar el conjunto de datos nuevo
#     df = pd.read_csv(input_data_path)

#     # Preprocesar los datos (escalar características)
#     X = df.drop(['quality', 'quality_label', 'quality_label_encoded'], axis=1)
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     # Realizar predicciones
#     y_pred = model.predict(X_scaled)
#     df['Predicción'] = y_pred

#     # Mostrar las predicciones
#     print("Predicciones realizadas:")
#     print(df[['quality_label', 'Predicción']])

#     # Guardar el archivo con las predicciones
#     df.to_csv('predicciones.csv', index=False)
#     print("Predicciones guardadas en 'predicciones.csv'")

# if __name__ == "__main__":
#     predecir('random_forest_model.pkl', 'data/new_data.csv')


import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Paso 1: Cargar el modelo entrenado
model = joblib.load('../notebooks/random_forest_model.pkl')

# Paso 2: Crear un DataFrame con los valores de entrada
input_data = {
    'fixed acidity': [7.4],
    'volatile acidity': [0.3],
    'citric acid': [0.4],
    'chlorides': [0.05],
    'total sulfur dioxide': [100],
    'density': [0.995],
    'sulphates': [0.65],
    'alcohol': [9.4]
}

# Crear un DataFrame a partir de los datos de entrada
df_input = pd.DataFrame(input_data)

# Paso 3: Escalar las características con el mismo escalador usado en el entrenamiento
scaler = StandardScaler()
df_input_scaled = scaler.fit_transform(df_input)

# Paso 4: Realizar la predicción con el modelo
predicted_class = model.predict(df_input_scaled)

# Paso 5: Mostrar el resultado
# Decodificar la predicción (si es necesario)
decoded_class = ['baja', 'media', 'alta']
predicted_label = decoded_class[predicted_class[0]]

print(f"La predicción de la calidad del vino es: {predicted_label}")
