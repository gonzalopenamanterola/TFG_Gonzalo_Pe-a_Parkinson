import os
import pandas as pd
import numpy as np
import tensorflow as tf
import keras._tf_keras.keras as keras
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras._tf_keras.keras.regularizers import l2
from keras._tf_keras.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def load_data(root_folder):
    data = []
    labels = []
    valid_folders = ["EPI", "EPG2019S", "CONTROLES", "AsG2019S-"]

    for main_folder in os.listdir(root_folder):
        if main_folder not in valid_folders:
            continue

        main_path = os.path.join(root_folder, main_folder)
        if not os.path.isdir(main_path):
            continue

        class_label = 0 if main_folder in ["CONTROLES", "AsG2019S-"] else 1

        for sub_folder in os.listdir(main_path):
            sub_path = os.path.join(main_path, sub_folder)
            if not os.path.isdir(sub_path):
                continue

            for target_folder in os.listdir(sub_path):
                if target_folder.startswith(("1 MINUTO MARCHA", "1MINUTO MARCHA", "15 M RAPIDO", "15M RAPIDO", "DOBLETASK")):
                    file_path = os.path.join(sub_path, target_folder, "AllData.csv")

                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path, header=0)
                        if "SourceFile" in df.columns:
                            df = df.drop(columns=["SourceFile"])
                        df = df.apply(pd.to_numeric, errors='coerce')
                        df.fillna(0, inplace=True)

                        if not df.empty:
                            data.append(df.values)
                            labels.append(class_label)

    print(f"Total de muestras cargadas: {len(data)}")
    return np.array(data, dtype=object), np.array(labels)

# Cargar datos
directory = r"C:\Users\gopem\TFG\Datos_Paper\Datos2018\Datos"
data, labels = load_data(directory)

if len(data) == 0:
    raise ValueError("No se encontraron datos válidos.")

MAX_TIMESTEPS = 5000
num_features = data[0].shape[1] if len(data) > 0 else 0
print(f"Limite de timesteps: {MAX_TIMESTEPS}, Num features: {num_features}")

data_trimmed = []
for sample in data:
    if len(sample) > MAX_TIMESTEPS:
        data_trimmed.append(sample[:MAX_TIMESTEPS])
    else:
        data_trimmed.append(np.pad(sample, ((0, MAX_TIMESTEPS - len(sample)), (0, 0)), mode='constant'))

data_padded = np.array(data_trimmed)

# Normalización
scaler = StandardScaler()
data_scaled = np.array([scaler.fit_transform(sample) for sample in data_padded])

# División de datos
X_train, X_test, y_train, y_test = train_test_split(data_scaled, labels, test_size=0.2, random_state=42)

print(f"Tamaño de X_train: {X_train.shape}, Tamaño de y_train: {y_train.shape}")
print(f"Tamaño de X_test: {X_test.shape}, Tamaño de y_test: {y_test.shape}")
print("Distribución de clases en entrenamiento:", np.bincount(y_train))

# Modelo con LSTM (no bidireccional)
model = Sequential([
    keras.Input(shape=(MAX_TIMESTEPS, num_features)),
    LSTM(128, return_sequences=True, kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.5), 
    LSTM(32, kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(8, activation='relu', kernel_regularizer=l2(0.001)),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, callbacks=[early_stop])

# Evaluación
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Pérdida: {loss}, Precisión: {accuracy}")

# Guardado
model.save("modelo_LSTM_v1.keras")
print("Modelo guardado como modelo_LSTM_v1.keras")

joblib.dump(scaler, "scaler_LSTM_v1.pkl")
print("Escalador guardado como scaler_LSTM_v1.pkl")
