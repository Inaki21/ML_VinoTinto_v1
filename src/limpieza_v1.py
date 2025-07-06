import pandas as pd
import os
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, StandardScaler

def limpiar_csv(input_csv_path, output_csv_path):
    # Crear la carpeta de salida si no existe
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    # Leer el archivo CSV original
    df = pd.read_csv(input_csv_path)
    
    # Verificar las columnas del DataFrame
    print("Columnas disponibles en el CSV:")
    print(df.columns)

    # Asegurarse de que la columna 'quality' existe
    if 'quality' not in df.columns:
        raise ValueError("'quality' no se encuentra en el archivo CSV. Verifique que el archivo de entrada sea el correcto.")

    # Crear la columna 'quality_label' si no está presente
    if 'quality_label' not in df.columns:
        df['quality_label'] = pd.cut(df['quality'], bins=[0, 5, 7, 10], labels=['baja', 'media', 'alta'])

    # Crear 'quality_label_encoded' si no está presente
    if 'quality_label_encoded' not in df.columns:
        label_encoder = LabelEncoder()
        df['quality_label_encoded'] = label_encoder.fit_transform(df['quality_label'])

    # Definir las características (X) y la variable objetivo (y)
    X = df.drop(['quality_label_encoded'], axis=1)  # No eliminar 'quality' ni 'quality_label'
    y = df['quality_label_encoded']

    # Balanceo de clases con SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    # Escalar las características
    scaler = StandardScaler()
    X_res_scaled = scaler.fit_transform(X_res)

    # Guardar el DataFrame procesado con el escalado
    df_processed = pd.DataFrame(X_res_scaled, columns=X.columns)
    df_processed['quality_label_encoded'] = y_res
    df_processed['quality'] = df['quality']  # Conservar la columna 'quality'
    df_processed['quality_label'] = df['quality_label']  # Conservar la columna 'quality_label'
    
    df_processed.to_csv(output_csv_path, index=False)
    print(f"CSV procesado guardado en: {output_csv_path}")

if __name__ == "__main__":
    limpiar_csv('../data/raw/winequality-red.csv', '../data/processed/vinotinto_v1.csv')
