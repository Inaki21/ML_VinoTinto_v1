import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluar_modelo(model_path, X_test, y_test):
    # Cargar el modelo previamente entrenado
    model = joblib.load(model_path)

    # Realizar predicciones
    y_pred = model.predict(X_test)

    # Imprimir el reporte de clasificación
    print("Evaluación del Modelo:")
    print(classification_report(y_test, y_pred))

    # Mostrar la matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.show()

def cargar_datos(input_csv_path):
    # Cargar el CSV procesado y preparar X y y para evaluación
    df = pd.read_csv(input_csv_path)
    
    # Definir las características (X) y la variable objetivo (y)
    X = df.drop(['quality_label_encoded'], axis=1)  # Eliminar las columnas de etiquetas
    y = df['quality_label_encoded']  # Usar 'quality_label_encoded' como objetivo

    return X, y

if __name__ == "__main__":
    # Cargar los datos de prueba y entrenamiento
    X_test, y_test = cargar_datos('../data/processed/vinotinto_v1.csv')
    
    # Cargar y evaluar el modelo
    evaluar_modelo('../modelos/random_forest_model_v1.pkl', X_test, y_test)

