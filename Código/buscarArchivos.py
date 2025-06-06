import os

# Define el directorio principal donde comienzas a buscar
root_dir = "C:\\Users\\gopem\\TFG\\Datos_Paper\\Datos de la marcha"

# Recorre todas las carpetas y subcarpetas
for root, dirs, files in os.walk(root_dir):
    # Filtra solo los archivos CSV
    csv_files = [file for file in files if file.endswith('.csv')]
    if csv_files:
        print(f'Archivos CSV encontrados en {root}: {csv_files}')
    else:
        print(f'No se encontraron archivos CSV en {root}')
