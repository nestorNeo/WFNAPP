import os
import sys
import subprocess
from pathlib import Path
from PyQt5.QtWidgets import QApplication
from src.ui.main_window import MainWindow

# Configurar rutas internas
base_path = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent))  # Base de PyInstaller
resources_path = base_path / 'resources'  # Carpeta de recursos empaquetada
bin_path = base_path / 'bin'  # Carpeta Graphviz/bin empaquetada

# Agregar bin al PATH
os.environ["PATH"] += os.pathsep + str(bin_path)
print(f"Graphviz/bin añadido al PATH interno: {bin_path}")

# Verificar si Graphviz está accesible
try:
    result = subprocess.run(['dot', '-V'], capture_output=True, text=True)
    print(f"Graphviz versión detectada: {result.stdout.strip()}")
except FileNotFoundError:
    print("No se encontró 'dot'. Asegúrate de que Graphviz/bin está configurado correctamente.")

if __name__ == "__main__":
    # Crear la aplicación de PyQt
    app = QApplication(sys.argv)

    # Aplicar estilos globales para la interfaz
    app.setStyleSheet("""
        QTextEdit {
            font-size: 24pt;
        }
        QLabel {
            font-size: 14pt;
        }
        QPushButton {
            font-size: 14pt;
        }
        QComboBox {
            font-size: 14pt;
        }
    """)

    # Crear y mostrar la ventana principal
    ventana_principal = MainWindow()
    ventana_principal.show()

    # Iniciar el bucle de eventos
    sys.exit(app.exec_())
