import os
import pandas as pd

class DataHandler:
    def __init__(self):
        self.valid_files = [
            "Caderas.csv", "Codos.csv", "Cuello.csv", "Hombros.csv",
            "Muñecas.csv", "Pelvis.csv", "Rodillas.csv", "Tobillos.csv", "Torso.csv"
        ]
        self.valid_prefixes = ["1 MINUTO MARCHA", "1MINUTO MARCHA", "15 M RAPIDO", "15M RAPIDO", "DOBLETASK"]

    def merge_csv_files(self, root_folder):
        for target_folder in os.listdir(root_folder):
            if not any(target_folder.startswith(prefix) for prefix in self.valid_prefixes):
                continue

            target_path = os.path.join(root_folder, target_folder)
            if not os.path.isdir(target_path) or os.path.exists(os.path.join(target_path, "AllData.csv")):
                continue

            dataframes = []
            for file_name in self.valid_files:
                file_path = os.path.join(target_path, file_name)
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    df = df.loc[:, ~df.columns.str.endswith('_X')]
                    df['SourceFile'] = file_name
                    dataframes.append(df)

            if dataframes:
                merged_df = pd.concat(dataframes, ignore_index=True)
                merged_df.to_csv(os.path.join(target_path, "AllData.csv"), index=False)

    def load_combined_data(self, folder_path):
        dfs = []
        for folder in os.listdir(folder_path):
            if any(folder.startswith(prefix) for prefix in self.valid_prefixes):
                file_path = os.path.join(folder_path, folder, "AllData.csv")
                if os.path.exists(file_path):
                    
                    df = pd.read_csv(file_path)

                    if "SourceFile" in df.columns:
                        df = df.drop(columns=["SourceFile"])

                    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

                    if not df.empty:
                        dfs.append(df)
        return pd.concat(dfs, ignore_index=True) if dfs else None
