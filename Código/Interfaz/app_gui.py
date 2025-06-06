import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from data_handler import DataHandler
from model_handler import ModelHandler

class AppGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Clasificación con Keras")
        self.root.geometry("800x500")

        self.folder_path = ""
        self.data_handler = DataHandler()
        self.model_handler = ModelHandler(
            "./Modelos/modelo_BiLSTM.keras",
            "./Modelos/scaler_BiLSTM.pkl"
        )

        self.create_widgets()

    def create_widgets(self):
        self.util_frame = tk.Frame(self.root)
        self.util_frame.pack(side=tk.LEFT, padx=10, pady=10)

        tk.Label(self.util_frame, text="Utilidades", font=("Arial", 12, "bold")).pack()
        tk.Button(self.util_frame, text="Seleccionar carpeta", command=self.select_folder, height=2, width=20).pack(pady=5)
        self.process_button = tk.Button(self.util_frame, text="Procesar datos", command=self.process_data, state=tk.DISABLED, height=2, width=20)
        self.process_button.pack(pady=5)
        self.classify_button = tk.Button(self.util_frame, text="Clasificar datos", command=self.classify_data, state=tk.DISABLED, height=2, width=20)
        self.classify_button.pack(pady=5)

        self.data_frame = tk.Frame(self.root)
        self.data_frame.pack(side=tk.LEFT, padx=20, pady=10, fill=tk.BOTH, expand=True)

        tk.Label(self.data_frame, text="Carpeta cargada", font=("Arial", 12, "bold")).pack()
        self.folder_display = tk.Label(self.data_frame, text="No seleccionada", relief=tk.SUNKEN, width=60, height=2)
        self.folder_display.pack(pady=5)

        tk.Label(self.data_frame, text="Resultados de la clasificación", font=("Arial", 12, "bold")).pack(pady=50)
        self.result_display = tk.Label(self.data_frame, text="", font=("Arial", 14, "bold"))
        self.result_display.pack()
        self.image_label = tk.Label(self.data_frame)
        self.image_label.pack()

        self.real_values_frame = tk.Frame(self.root)
        self.real_values_frame.pack(side=tk.LEFT, padx=10, pady=10)

        tk.Label(self.real_values_frame, text="Valores reales", font=("Arial", 12, "bold")).pack()
        self.real_values_display = tk.Text(self.real_values_frame, height=10, width=40)
        self.real_values_display.pack()

    def select_folder(self):
        self.folder_path = filedialog.askdirectory()
        if self.folder_path:
            self.folder_display.config(text=self.folder_path)
            self.process_button.config(state=tk.NORMAL)

    def process_data(self):
        try:
            self.data_handler.merge_csv_files(self.folder_path)
            messagebox.showinfo("Éxito", "Datos fusionados correctamente.")
            self.classify_button.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"Error al procesar datos: {e}")


    def classify_data(self):
        try:
            df = self.data_handler.load_combined_data(self.folder_path)
            if df is None:
                messagebox.showerror("Error", "No se encontraron datos válidos para clasificar.")
                return

            segments = self.model_handler.prepare_data(df)
            predictions, prob, rounded = self.model_handler.predict(segments)
            label = "Sano" if rounded == 0 else "Parkinson"

            self.result_display.config(text=label)
            image_path = f"./Interfaz/Imagenes/{'okey' if label == 'Sano' else 'bad'}.png"
            img = Image.open(image_path).resize((100, 100))
            self.image = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.image)
            self.image_label.image = self.image

            self.real_values_display.delete("1.0", tk.END)
            self.real_values_display.insert(tk.END, f"Probabilidades de los segmentos: {predictions}\n")
            self.real_values_display.insert(tk.END, f"Predicción media: {prob:.4f}\n")
            self.real_values_display.insert(tk.END, f"Predicción redondeada: {int(rounded)}\n")
            self.real_values_display.insert(tk.END, f"Resultado: {label}\n")
        except Exception as e:
            messagebox.showerror("Error", f"Error al clasificar datos: {e}")

