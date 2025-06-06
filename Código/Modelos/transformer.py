import os
import pandas as pd
import numpy as np
import tensorflow as tf
import keras._tf_keras.keras as keras
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import Input, Dense, Dropout, LayerNormalization
from keras._tf_keras.keras.layers import MultiHeadAttention, GlobalAveragePooling1D
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

            dfs_to_concat = []

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
                            dfs_to_concat.append(df)

            if dfs_to_concat:
                combined_df = pd.concat(dfs_to_concat, ignore_index=True)
                data.append(combined_df.values)
                labels.append(class_label)

    print(f"Total de muestras cargadas: {len(data)}")
    return np.array(data, dtype=object), np.array(labels)

# Cargar y procesar los datos como ya haces
directory = r"C:\Users\gopem\TFG\Datos_Paper\Datos2018\Datos"
data, labels = load_data(directory)

if len(data) == 0:
    raise ValueError("No se encontraron datos válidos.")

MAX_TIMESTEPS = 1000
num_features = data[0].shape[1] if len(data) > 0 else 0

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

# División
X_train, X_test, y_train, y_test = train_test_split(data_scaled, labels, test_size=0.2, random_state=42)

print(f"Tamaño de X_train: {X_train.shape}, Tamaño de X_test: {X_test.shape}")
print("Distribución de clases:", np.bincount(y_train))

def transformer_block(inputs, num_heads, ff_dim, dropout):
    # Multi-Head Attention
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(inputs, inputs)
    attention_output = Dropout(dropout)(attention_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attention_output)


    # Feed Forward
    ff_output = Dense(ff_dim, activation="relu")(out1)
    ff_output = Dense(x.shape[-1])(ff_output)  # ← proyecta de vuelta a misma dimensión que out1
    ff_output = Dropout(dropout)(ff_output)
    
    return LayerNormalization(epsilon=1e-6)(out1 + ff_output)
# Input
inputs = Input(shape=(MAX_TIMESTEPS, num_features))

# Bloques Transformer
inputs = keras.Input(shape=(MAX_TIMESTEPS, num_features))  # 44
x = Dense(128)(inputs)  # Convertimos a 128 dimensiones desde el inicio
x = transformer_block(x, num_heads=4, ff_dim=128, dropout=0.5)

# Pooling global + clasificador
x = GlobalAveragePooling1D()(x)
x = Dropout(0.5)(x)
x = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=outputs)

# Compilar
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, callbacks=[early_stop])

# Evaluar
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Pérdida: {loss}, Precisión: {accuracy}")

# Guardar
model.save("modelo_transformer_v1.keras")
joblib.dump(scaler, "scaler_transformer_v1.pkl")
print("Modelo y escalador guardados.")
