import sys
import os
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QFileDialog,
    QLabel, QMenuBar, QAction, QFontDialog, QMessageBox, QScrollArea, QApplication
)
from PyQt5.QtCore import Qt, QPropertyAnimation
from PyQt5.QtGui import QIcon, QPixmap, QTransform, QFont, QColor, QPalette
from fpdf import FPDF
from PIL import Image
from src.core.petrinet import (
    get_traces, analyze_precedence, analyze_with_transitive_closure, analyze_and_construct_wfn,
    simplify_extended_relation
)
from src.visualization.visualizer import (
    visualize_sequences, visualize_precedence, visualize_transitive_closure,
    visualize_simplified_relation, visualize_wfn
)


# Función para aplicar un tema oscuro
def apply_dark_theme(app):
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    app.setPalette(palette)


class ImageViewer(QWidget):
    """Widget para visualizar imágenes con opciones de zoom"""

    def __init__(self, image_path):
        super().__init__()
        self.setWindowTitle("Image Viewer")

        self.image_path = image_path
        self.zoom_factor = 1.0

        # Configuración del layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Scroll para la imagen
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.layout.addWidget(self.scroll_area)

        # Configuración de la imagen en el QLabel
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.pixmap = QPixmap(self.image_path)
        self.image_label.setPixmap(self.pixmap)
        self.scroll_area.setWidget(self.image_label)
        self.adjust_window_size()

        # Botones de zoom
        self.zoom_in_button = QPushButton("Zoom In")
        self.zoom_in_button.clicked.connect(self.zoom_in)
        self.layout.addWidget(self.zoom_in_button)

        self.zoom_out_button = QPushButton("Zoom Out")
        self.zoom_out_button.clicked.connect(self.zoom_out)
        self.layout.addWidget(self.zoom_out_button)

    def adjust_window_size(self):
        image_width = self.pixmap.width()
        image_height = self.pixmap.height()
        self.setMinimumSize(400, 300)
        self.resize(image_width, image_height)

    def zoom_in(self):
        self.zoom_factor *= 1.25
        self.apply_zoom()

    def zoom_out(self):
        self.zoom_factor /= 1.25
        self.apply_zoom()

    def apply_zoom(self):
        transform = QTransform().scale(self.zoom_factor, self.zoom_factor)
        scaled_pixmap = self.pixmap.transformed(transform, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.resize(scaled_pixmap.size())
        self.scroll_area.setWidget(self.image_label)


class MainWindow(QMainWindow):
    """Main window with a modern interface"""

     # Agregar función para manejar rutas de recursos
    def get_resource_path(self, relative_path):
        """Obtiene la ruta absoluta a los recursos"""
        try:
            if hasattr(sys, '_MEIPASS'):
                path = os.path.join(sys._MEIPASS, relative_path)
            else:
                path = os.path.join(os.path.abspath("."), relative_path)
                
            # Si es un ícono y no existe, intentar con la versión .ico
            if not os.path.exists(path) and relative_path.endswith(('.jpg', '.png')):
                ico_path = path.rsplit('.', 1)[0] + '.ico'
                if os.path.exists(ico_path):
                    return ico_path
                    
            return path
        except Exception as e:
            print(f"Error al obtener recurso {relative_path}: {str(e)}")
            return None

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Synthesis of WFN from Reduced Event Logs")
        self.setGeometry(100, 100, 1000, 600)
        
        # Set modern icon
        icon_path = self.get_resource_path("resources/Cinvestavicono.jpg")
        self.setWindowIcon(QIcon(icon_path))

        # Initialize necessary attributes
        self.images = []  # List to store generated images
        self.output_label = QLabel()  # Label to display output messages
        self.output_label.setAlignment(Qt.AlignCenter)
        self.image_save_directory = os.path.expanduser("~")  # Default folder for saving images
        self.last_pdf_path = None  # Track the last saved PDF path

        # Modernized style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2e3440;
            }
            QTextEdit {
                background-color: #3b4252;
                color: #d8dee9;
                border: 1px solid #4c566a;
                padding: 10px;
                border-radius: 8px;
                font-family: 'Segoe UI', sans-serif;
                font-size: 30px;
            }
            QPushButton {
                background-color: #5e81ac;
                color: white;
                border-radius: 8px;
                padding: 8px 16px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #81a1c1;
            }
            QLabel {
                color: #88c0d0;
                font-size: 16px;
            }
        """)

        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Logo and title
        top_layout = QHBoxLayout()
        logo_label = QLabel(self)
        logo_pixmap = QPixmap(self.get_resource_path("resources/Cinvestav.jpg")).scaled(80, 80, Qt.KeepAspectRatio)
        logo_label.setPixmap(logo_pixmap)
        title_label = QLabel("Synthesis of WFN from Reduced Event Logs")
        title_label.setStyleSheet("font-size: 22px; font-weight: bold; color: #eceff4;")
        title_label.setAlignment(Qt.AlignCenter)
        top_layout.addWidget(logo_label)
        top_layout.addWidget(title_label)
        main_layout.addLayout(top_layout)

        # Content layout
        content_layout = QHBoxLayout()
        
        # Left column: text box
        text_layout = QVBoxLayout()
        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText("Enter traces here (e.g., X A B C D Y). Use a space between events and start each new trace on a separate line. Example:\n\nx a b c d y\nw e c b f z")
        text_layout.addWidget(self.input_text)
        content_layout.addLayout(text_layout)

        # Right column: analysis buttons
        button_layout = QVBoxLayout()
        
        # Button to generate PDF
        pdf_button = self.create_colored_button("Generate PDF", "#dc3545", self.generate_pdf)
        button_layout.addWidget(pdf_button)

        # Analysis buttons
        analysis_buttons = [
                ("Precedence Relation", self.process_precedence_with_iterations),
                ("Precedence Relation Without Iterations", self.process_precedence_without_iterations),
                ("Extended Transitive Closure Union", self.process_transitive_closure),
                ("Simplified Relation", self.process_simplified_relation),
                ("Workflow Net (WFN)", self.process_wfn),
                ("Workflow Net (WFN) Final Result", self.process_wfn_with_iterations),
                ("Show the Entire Process", self.process_all)
            ]

        for label, function in analysis_buttons:
            button = self.create_colored_button(label, "#5e81ac", function)
            button_layout.addWidget(button)

        # Add output label to button layout
        button_layout.addWidget(self.output_label)

        # Add content layout to main layout
        content_layout.addLayout(button_layout)
        main_layout.addLayout(content_layout)

        # Create menu options
        self.create_menu_bar()

    def create_colored_button(self, text, color, callback=None):
        """Creates a styled button"""
        button = QPushButton(text)
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border-radius: 8px;
            }}
            QPushButton:hover {{
                background-color: {self.darken_color(color, 0.1)};
            }}
        """)
        if callback:
            button.clicked.connect(callback)
        return button

    def darken_color(self, color, factor):
        """Darkens a hex color"""
        color = color.lstrip('#')
        rgb = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        dark_rgb = tuple(int(c * (1 - factor)) for c in rgb)
        return '#{:02x}{:02x}{:02x}'.format(*dark_rgb)

    def create_menu_bar(self):
        """Creates the menu bar"""
        menu_bar = self.menuBar()

        # File Menu
        file_menu = menu_bar.addMenu("File")
        
        open_action = QAction("Open Trace File", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.load_file)
        file_menu.addAction(open_action)

        select_directory_action = QAction("Select Folder to Save Images", self)
        select_directory_action.triggered.connect(self.select_image_directory)
        file_menu.addAction(select_directory_action)

        open_pdf_location_action = QAction("Open Last PDF Location", self)
        open_pdf_location_action.triggered.connect(self.open_last_pdf_location)
        file_menu.addAction(open_pdf_location_action)

        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Help Menu
        help_menu = menu_bar.addMenu("Help")
        help_action = QAction("Instructions", self)
        help_action.setShortcut("F1")
        help_action.triggered.connect(self.show_help)
        help_menu.addAction(help_action)

    def load_file(self):
        """Opens a trace file"""
        file_name, _ = QFileDialog.getOpenFileName(self, "Open trace file", "", "Text files (*.txt)")
        if file_name:
            with open(file_name, 'r') as file:
                self.input_text.setPlainText(file.read())

    def select_image_directory(self):
        """Selects the folder to save images"""
        directory = QFileDialog.getExistingDirectory(self, "Select Directory", self.image_save_directory)
        if directory:
            self.image_save_directory = directory
            QMessageBox.information(self, "Directory Selected", f"Images will be saved to: {self.image_save_directory}")

    def open_last_pdf_location(self):
        """Opens the folder where the last PDF was saved"""
        if self.last_pdf_path and os.path.exists(self.last_pdf_path):
            os.startfile(self.last_pdf_path)
        else:
            QMessageBox.warning(self, "No PDF Found", "No PDF has been saved yet.")

    def show_help(self):
        """Displays instructions"""
        instructions = (
            "Instructions:\n\n"
            "1. Open Trace File: Load a file with event traces to analyze.\n"
            "2. Select Folder to Save Images: Choose a folder to save generated images.\n"
            "3. Open Last PDF Location: Open the folder where the last PDF was saved.\n"
            "4. Generate PDF: Generate a PDF containing all generated images.\n"
            "5. Analysis Buttons: Choose an analysis type to generate specific images."
        )
        QMessageBox.information(self, "Help", instructions)

    # Analysis functions with full logic
    def process_precedence_with_iterations(self):
        self.process_single("Precedence relation with iterations")

    def process_precedence_without_iterations(self):
        self.process_single("Precedence relation without iterations")

    def process_transitive_closure(self):
        self.process_single("Precedence relation with transitive closure")

    def process_simplified_relation(self):
        self.process_single("Simplified relation")

    def process_wfn(self):
        self.process_single("Workflow Net (WFN)")

    def process_wfn_with_iterations(self):
        self.process_single("Workflow Net (WFN) with iterations")

    def process_single(self, analysis_type):
        input_text = self.input_text.toPlainText()
        traces = get_traces(input_text)

        if not traces:
            self.output_label.setText("Error: No valid traces were provided.")
            return
        
        try:
            image_path = None
            iterations_info = ""
            
            if analysis_type == "Sequences":
                image_path = visualize_sequences(traces)
                
            elif analysis_type == "Precedence relation with iterations":
                precedence_with_iterations, _, _, processed_traces, iterations = analyze_precedence(traces)
                image_path = visualize_precedence(precedence_with_iterations, processed_traces, analysis_type.replace(" ", "_").lower(), include_loops=False)
                if iterations:
                    iterations_info = "\nIteraciones detectadas:\n"
                    for it in iterations:
                        iterations_info += f"- De '{it.start_event}' a '{it.end_event}' con secuencia {it.sequence}\n"
                    
            elif analysis_type == "Precedence relation without iterations":
                _, precedence_without_iterations, _, processed_traces, iterations = analyze_precedence(traces)
                image_path = visualize_precedence(precedence_without_iterations, processed_traces, analysis_type.replace(" ", "_").lower(), include_loops=True)
                
            elif analysis_type == "Precedence relation with transitive closure":
                closure, processed_traces, precedence_without_iterations = analyze_with_transitive_closure(traces)
                image_path = visualize_transitive_closure(processed_traces, closure, precedence_without_iterations, "transitive_closure")
                
            elif analysis_type == "Simplified relation":
                closure, processed_traces, precedence_without_iterations = analyze_with_transitive_closure(traces)
                simplified_relation, bridges = simplify_extended_relation(precedence_without_iterations, closure, processed_traces)
                image_path = visualize_simplified_relation(processed_traces, simplified_relation, bridges, precedence_without_iterations, "simplified_relation")
                
            elif analysis_type == "Workflow Net (WFN)":
                wfn, _, _, bridges = analyze_and_construct_wfn(traces, include_iterations=False)
                image_path = visualize_wfn(wfn, bridges, "workflow_net_without_iterations")
                
            elif analysis_type == "Workflow Net (WFN) with iterations":
                wfn, _, _, bridges = analyze_and_construct_wfn(traces, include_iterations=True)
                image_path = visualize_wfn(wfn, bridges, "workflow_net_with_iterations")
                _, _, _, processed_traces, iterations = analyze_precedence(traces)
                if iterations:
                    iterations_info = "\nIteraciones detectadas:\n"
                    for it in iterations:
                        iterations_info += f"- De '{it.start_event}' a '{it.end_event}' con secuencia {it.sequence}\n"
            
            if image_path:
                if (analysis_type, image_path) not in self.images:
                    self.images.append((analysis_type, image_path))
                self.show_image(image_path, analysis_type)
                if iterations_info:
                    self.output_label.setText(f"Imagen generada: {image_path}\n{iterations_info}")
                else:
                    self.output_label.setText(f"Imagen generada: {image_path}")

        except Exception as e:
            error_message = f"Error: {str(e)}"
            self.output_label.setText(error_message)

    def process_all(self):
        self.images = []

        analysis_types = [
            "Sequences",
            "Precedence relation with iterations",
            "Precedence relation without iterations",
            "Precedence relation with transitive closure",
            "Simplified relation",
            "Workflow Net (WFN)",
            "Workflow Net (WFN) with iterations"
        ]

        try:
            for analysis_type in analysis_types:
                self.process_single(analysis_type)

            if self.images:
                self.output_label.setText(f"{len(self.images)} images generated. Click 'Generate PDF' to create the PDF.")
                self.show_all_images()

        except Exception as e:
            error_message = f"Error processing all: {str(e)}"
            self.output_label.setText(error_message)

    def show_image(self, image_path, title):
        if not hasattr(self, 'image_viewers'):
            self.image_viewers = []

        existing_viewer = next((viewer for viewer in self.image_viewers if viewer.windowTitle() == title), None)
        if existing_viewer:
            existing_viewer.close()
            self.image_viewers.remove(existing_viewer)

        image_viewer = ImageViewer(image_path)
        image_viewer.setWindowTitle(title)
        image_viewer.show()

        self.image_viewers.append(image_viewer)

    def show_all_images(self):
        if not self.images:
            self.output_label.setText("No images to show. Process the data first.")
            return

        if hasattr(self, 'image_viewers'):
            for viewer in self.image_viewers:
                viewer.close()
            self.image_viewers.clear()

        for title, image_path in self.images:
            self.show_image(image_path, title)

        self.output_label.setText(f"{len(self.images)} images shown.")

    # PDF generation method
    def generate_pdf(self):
        if not self.images:
            self.output_label.setText("No images to include in the PDF. Process the data first.")
            return

        pdf = FPDF(orientation='L')
        pdf.set_auto_page_break(auto=True, margin=15)

        logo_path = self.get_resource_path("resources/Cinvestav.jpg")

        for title, image_path in self.images:
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, title, 0, 1, 'C')

            if os.path.exists(logo_path):
                pdf.image(logo_path, 10, 10, 30)

            try:
                img_width, img_height = Image.open(image_path).size
                page_width = pdf.w - 20
                page_height = pdf.h - 40
                width_ratio = page_width / img_width
                height_ratio = page_height / img_height
                scale_factor = min(width_ratio, height_ratio)

                new_width = img_width * scale_factor
                new_height = img_height * scale_factor

                x = (pdf.w - new_width) / 2
                y = (pdf.h - new_height) / 2 + 10
                pdf.image(image_path, x=x, y=y, w=new_width, h=new_height)

            except Exception as e:
                print(f"Error adding image {image_path}: {e}")

        pdf_file_name, _ = QFileDialog.getSaveFileName(self, "Save PDF", "", "PDF Files (*.pdf)")
        if pdf_file_name:
            pdf.output(pdf_file_name)
            self.last_pdf_path = os.path.dirname(pdf_file_name)
            self.output_label.setText(f"PDF saved to: {pdf_file_name}")
        else:
            self.output_label.setText("PDF save canceled.")