import os
import pandas as pd

def merge_csv_files(root_folder):
    file_list = ["Caderas.csv", "Codos.csv", "Cuello.csv", "Hombros.csv", "Muñecas.csv", 
                 "Pelvis.csv", "Rodillas.csv", "Tobillos.csv", "Torso.csv"]
    
    prefixes = ["1 MINUTO MARCHA", "1MINUTO MARCHA", "15 M RAPIDO", "15M RAPIDO", "DOBLETASK"]
    
    # Recorrer las carpetas principales
    for main_folder in os.listdir(root_folder):
        main_path = os.path.join(root_folder, main_folder)
        
        if not os.path.isdir(main_path):
            continue
        
        # Recorrer las subcarpetas dentro de cada carpeta principal
        for sub_folder in os.listdir(main_path):
            sub_path = os.path.join(main_path, sub_folder)
            
            if not os.path.isdir(sub_path):
                continue
            
            # Procesar las carpetas específicas
            for target_folder in os.listdir(sub_path):
                if not any(target_folder.startswith(prefix) for prefix in prefixes):
                    continue
                
                target_path = os.path.join(sub_path, target_folder)
                
                if not os.path.isdir(target_path):
                    continue
                
                # Lista para almacenar los DataFrames
                dataframes = []
                
                # Leer los archivos CSV y fusionarlos
                for file_name in file_list:
                    file_path = os.path.join(target_path, file_name)
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path)
                        df = df.loc[:, ~df.columns.str.endswith('_X')] # Eliminar columnas que terminan en '_X'
                        df['SourceFile'] = file_name  # Agregar columna para identificar el origen
                        dataframes.append(df)
                
                # Si hay archivos para fusionar, guardamos el nuevo CSV
                if dataframes:
                    merged_df = pd.concat(dataframes, ignore_index=True)

                    if 'Tiempo' in merged_df.columns:
                        merged_df.drop(columns=['Tiempo'], inplace=True)

                    merged_df.to_csv(os.path.join(target_path, "AllData.csv"), index=False)
                    print(f"Fusionado: {os.path.join(target_path, 'AllData.csv')}")

# Ruta principal
directory = r"C:\\Users\\gopem\\TFG\\Datos_Paper\\Datos2018\\Datos"
merge_csv_files(directory)
