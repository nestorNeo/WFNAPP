from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class ImageViewer(QWidget):
    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Visualización')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        image_label = QLabel()
        pixmap = QPixmap(self.image_path)
        image_label.setPixmap(pixmap)
        image_label.setAlignment(Qt.AlignCenter)

        scroll_area.setWidget(image_label)
        layout.addWidget(scroll_area)

        self.setLayout(layout)
