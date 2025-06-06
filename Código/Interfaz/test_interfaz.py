import unittest
from unittest.mock import patch, MagicMock
import tkinter as tk
from app_gui import AppGUI

class TestAppGUI(unittest.TestCase):
    def setUp(self):
        # Crear una raíz oculta para evitar mostrar la ventana real
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = AppGUI(self.root)

    def tearDown(self):
        self.root.destroy()

    @patch('app_gui.filedialog.askdirectory')
    def test_select_folder_updates_ui(self, mock_askdirectory):
        # Simular selección de carpeta
        mock_askdirectory.return_value = "/ruta/de/prueba"
        self.app.select_folder()

        self.assertEqual(self.app.folder_path, "/ruta/de/prueba")
        self.assertEqual(self.app.folder_display.cget("text"), "/ruta/de/prueba")
        self.assertEqual(self.app.process_button['state'], tk.NORMAL)

    @patch('app_gui.messagebox.showinfo')
    @patch('app_gui.DataHandler.merge_csv_files')
    def test_process_data_success(self, mock_merge, mock_info):
        self.app.folder_path = "/ruta/falsa"
        self.app.process_data()

        mock_merge.assert_called_once_with("/ruta/falsa")
        mock_info.assert_called_once_with("Éxito", "Datos fusionados correctamente.")
        self.assertEqual(self.app.classify_button['state'], tk.NORMAL)

    @patch('app_gui.messagebox.showerror')
    @patch('app_gui.DataHandler.load_combined_data', return_value=None)
    def test_classify_data_no_data(self, mock_load, mock_error):
        self.app.folder_path = "/ruta/invalida"
        self.app.classify_data()

        mock_error.assert_called_once_with("Error", "No se encontraron datos válidos para clasificar.")
        self.assertEqual(self.app.result_display.cget("text"), "")

    @patch('app_gui.ImageTk.PhotoImage')
    @patch('app_gui.Image.open')
    @patch('app_gui.ModelHandler.predict', return_value=([[0.2], [0.4]], 0.3, 0.0))
    @patch('app_gui.DataHandler.load_combined_data')
    def test_classify_data_valid(self, mock_load, mock_predict, mock_image_open, mock_photoimage):
        df_mock = MagicMock()
        mock_load.return_value = df_mock
        self.app.model_handler.prepare_data = MagicMock(return_value=["segmento"])
        
        # Mock necesario para que PhotoImage no dé error al asignarse a un widget
        fake_image = MagicMock(name="FakePhotoImage")
        mock_photoimage.return_value = fake_image
        mock_image_open.return_value.resize.return_value = MagicMock(name="FakeResizedImage")

        # También mockeamos `config` del image_label para evitar errores de GUI
        self.app.image_label.config = MagicMock()

        self.app.folder_path = "/ruta/valida"
        self.app.classify_data()

        mock_predict.assert_called()
        self.assertIn("Sano", self.app.result_display.cget("text"))

    @patch('app_gui.messagebox.showerror')
    @patch('app_gui.DataHandler.merge_csv_files', side_effect=Exception("Error inesperado"))
    def test_process_data_exception(self, mock_merge, mock_error):
        self.app.folder_path = "/ruta/falsa"
        try:
            self.app.process_data()
        except Exception:
            self.fail("process_data() lanzó una excepción no manejada")

        mock_merge.assert_called_once()
        mock_error.assert_called_once_with("Error", "Error al procesar datos: Error inesperado")


    @patch('app_gui.messagebox.showerror')
    @patch('app_gui.ModelHandler.predict', side_effect=Exception("Modelo roto"))
    @patch('app_gui.DataHandler.load_combined_data')
    def test_classify_data_exception(self, mock_load, mock_predict, mock_error):
        df_mock = MagicMock()
        mock_load.return_value = df_mock
        self.app.model_handler.prepare_data = MagicMock(return_value=["segmento"])

        # Evitar error por imagen al hacer `.config(image=...)`
        self.app.image_label.config = MagicMock()

        self.app.folder_path = "/ruta/valida"
        try:
            self.app.classify_data()
        except Exception:
            self.fail("classify_data() lanzó una excepción no manejada")

        mock_predict.assert_called_once()
        mock_error.assert_called_once_with("Error", "Error al clasificar datos: Modelo roto")


    @patch('app_gui.messagebox.showinfo')
    @patch('app_gui.DataHandler.merge_csv_files')
    def test_classify_button_enabled_after_process(self, mock_merge, mock_info):
        self.app.folder_path = "/ruta/falsa"
        self.assertEqual(self.app.classify_button['state'], tk.DISABLED)
        self.app.process_data()
        self.assertEqual(self.app.classify_button['state'], tk.NORMAL)


    def test_initial_ui_state(self):
        self.assertEqual(self.app.process_button['state'], tk.DISABLED)
        self.assertEqual(self.app.classify_button['state'], tk.DISABLED)
        self.assertEqual(self.app.folder_display.cget("text"), "No seleccionada")
        self.assertEqual(self.app.result_display.cget("text"), "")

if __name__ == '__main__':
    unittest.main()
