import numpy as np
import joblib
from keras._tf_keras.keras.models import load_model

class ModelHandler:
    def __init__(self, model_path, scaler_path):
        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.sequence_length = 5000
        self.step_size = 2500

    def prepare_data(self, df):
        data = self.scaler.transform(df.values)
        data = np.reshape(data, (1, data.shape[0], data.shape[1]))

        if data.shape[1] < self.sequence_length:
            padded = np.pad(data, ((0, 0), (0, self.sequence_length - data.shape[1]), (0, 0)), mode='constant')
            return [padded]
        else:
            return [
                data[:, i:i + self.sequence_length, :]
                for i in range(0, data.shape[1] - self.sequence_length + 1, self.step_size)
            ]

    def predict(self, segments):
        segments = np.squeeze(np.array(segments), axis=1)
        preds = self.model.predict(segments)

        final_proba = np.mean(preds)  
        
        umbral = 0.7
        prediction = int(final_proba >= umbral)

        return preds, np.mean(final_proba), prediction
