import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import os

def entrenar_modelo(input_csv_path, model_output_path):
    # Leer los datos procesados
    df = pd.read_csv(input_csv_path)

    # Verificar las columnas del DataFrame
    print(df.columns)

    # Definir las características (X) y la variable objetivo (y)
    X = df.drop(['quality_label_encoded'], axis=1)  # Eliminar la columna de la etiqueta codificada
    y = df['quality_label_encoded']  # Usar 'quality_label_encoded' como objetivo

    # Balanceo de clases con SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

    # Escalar las características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Inicializar y entrenar el modelo Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
    rf.fit(X_train_scaled, y_train)

    # Validación cruzada
    cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"Validación Cruzada (Accuracy): {cv_scores}")
    print(f"Promedio de Accuracy: {cv_scores.mean()}")

    # Crear la carpeta de salida si no existe
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)

    # Guardar el modelo entrenado
    joblib.dump(rf, model_output_path)
    print(f"Modelo entrenado guardado en: {model_output_path}")

if __name__ == "__main__":
    entrenar_modelo('../data/processed/vinotinto_v1.csv', '../modelos/random_forest_model_v1.pkl')
