import pandas as pd
import os

def leer_csv_estructura(directorio_padre):
    """
    Lee archivos CSV ('eventos.csv' y 'LocalAccel.csv') dentro de subdirectorios organizados en niveles:
    - Familia (directorio padre)
        - Hijos (subdirectorios de Familia)
            - Condiciones (subdirectorios de cada hijo: '1minuto', '15minutos', 'DobleTarea')

    :param directorio_padre: Ruta del directorio padre donde están los subdirectorios (hijos).
    :return: Diccionario con estructura { hijo: { subcarpeta: [DataFrames] } }.
    """
    archivos_deseados = ["eventos.csv", "LocalAccel.csv"]
    diccionario_df = {}

    # Verificar que el directorio padre existe
    if os.path.exists(directorio_padre):
        # Iterar sobre cada hijo dentro de Familia
        for hijo in os.listdir(directorio_padre):
            #print("Soy el hijo: " + str(hijo))
            hijo_path = os.path.join(directorio_padre, hijo)

            # Verificar si es un directorio
            if os.path.isdir(hijo_path):
                diccionario_df[hijo] = {}  # Crear entrada para el hijo

                # Iterar sobre cada subcarpeta dentro del hijo (1minuto, 15minutos, DobleTarea)
                for subcarpeta in os.listdir(hijo_path):
                    subcarpeta_path = os.path.join(hijo_path, subcarpeta)

                    # Verificar si es un directorio
                    if os.path.isdir(subcarpeta_path):
                        lista_df = []  # Lista para los DataFrames de esta subcarpeta

                        # Leer los archivos dentro de la subcarpeta
                        for archivo in os.listdir(subcarpeta_path):
                            if archivo in archivos_deseados:
                                ruta_completa = os.path.join(subcarpeta_path, archivo)
                                try:
                                    df = pd.read_csv(ruta_completa)
                                    lista_df.append(df)
                                except Exception as e:
                                    print(f"Error al leer el archivo {archivo} en {subcarpeta} de {hijo}: {e}")

                        # Si se encontraron archivos en la subcarpeta, se guardan en el diccionario
                        if lista_df:
                            diccionario_df[hijo][subcarpeta] = lista_df

    else:
        print(f"El directorio padre {directorio_padre} no existe.")

    return diccionario_df if diccionario_df else None

def comprobar():
    for hijo, subcarpetas in datos.items():
        print(f"Hijo: {hijo}")
        for subcarpeta, dataframes in subcarpetas.items():
            print(f"Subcarpeta: {subcarpeta} - {len(dataframes)} archivos leídos")



# Ruta al directorio "Familia"
directorio_padre = "C:\\Users\\gopem\\TFG\\Datos_Paper\\Datos de la marcha\\AsG2019S-"

# Leer los archivos estructurados
datos = leer_csv_estructura(directorio_padre)
comprobar()





