import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from keras._tf_keras.keras.models import load_model

def load_asg2019splus_data(root_folder):
    """
    Carga los datos de la carpeta AsG2019S+ para su evaluación.
    Etiqueta con '1' los sujetos especiales MZB, IOC y FCM (Parkinson).
    Etiqueta con '0' los demás sujetos (Sano).
    """
    data = []
    labels = []
    subject_names = []
    target_folder = "AsG2019S+"
    subjects_with_parkinson = ["MZB", "IOC", "FCM"]

    main_path = os.path.join(root_folder, target_folder)
    if not os.path.isdir(main_path):
        print(f"La carpeta {target_folder} no existe en {root_folder}")
        return np.array(data, dtype=object), np.array(labels), []

    for subject_folder in os.listdir(main_path):
        subject_path = os.path.join(main_path, subject_folder)
        if not os.path.isdir(subject_path):
            continue

        class_label = 1 if subject_folder in subjects_with_parkinson else 0
        dfs_to_concat = []

        for folder in os.listdir(subject_path):
            file_path = os.path.join(subject_path, folder, "AllData.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                if "SourceFile" in df.columns:
                    df = df.drop(columns=["SourceFile"])
                df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
                dfs_to_concat.append(df)

        if dfs_to_concat:
            combined_df = pd.concat(dfs_to_concat, ignore_index=True)
            data.append(combined_df.values)
            labels.append(class_label)
            subject_names.append(subject_folder)
        else:
            print(f"Advertencia: No se encontraron datos válidos para el sujeto {subject_folder}")

    return np.array(data, dtype=object), np.array(labels), subject_names


if __name__ == "__main__":
    # Configuración de rutas
    root_folder =  r"C:\Users\gopem\TFG\Datos_Paper\Datos2018\Datos"
    model_path = "./Modelos/modelo_BiLSTM.keras"
    scaler_path ="./Modelos/scaler_BiLSTM.pkl"

    # Cargar modelo y scaler
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    # Cargar datos
    data, labels, subject_names = load_asg2019splus_data(root_folder)

    y_true = []
    y_pred = []
    y_pred_proba = []

    for i, (sample, label, name) in enumerate(zip(data, labels, subject_names)):
        print(f"\n--- Evaluando sujeto: {name} ---")

        # Normalizar datos
        sample_scaled = scaler.transform(sample)
        # Redimensionar para el modelo
        sample_scaled = np.reshape(sample_scaled, (1, sample_scaled.shape[0], sample_scaled.shape[1]))

        # Fragmentación en ventanas
        sequence_length = 5000
        step_size = 2500
        data_splitted = []

        if sample_scaled.shape[1] < sequence_length:
            padded = np.pad(sample_scaled, ((0,0), (0, sequence_length - sample_scaled.shape[1]), (0,0)), mode='constant')
            data_splitted = [padded]
        else:
            for j in range(0, sample_scaled.shape[1] - sequence_length + 1, step_size):
                data_splitted.append(sample_scaled[:, j:j+sequence_length, :])

        data_splitted = np.squeeze(np.array(data_splitted), axis=1)
        proba = model.predict(data_splitted)

        final_proba = np.mean(proba)  
        
        umbral = 0.7
        prediction = int(final_proba >= umbral)

        y_true.append(label)
        y_pred.append(prediction)
        y_pred_proba.append(final_proba)

        print(f"Probabilidad media: {final_proba:.4f} | Predicción: {'Parkinson' if prediction else 'Sano'} | Etiqueta real: {'Parkinson' if label else 'Sano'}")
