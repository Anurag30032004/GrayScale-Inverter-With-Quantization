import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QSlider
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QPalette, QColor


class ImageConverter(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image to Grayscale Converter (NumPy Only)")
        self.setMinimumSize(900, 600)

        # Light modern background
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(245, 245, 245))
        self.setPalette(palette)

        # === Main Layout ===
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # --- Top controls layout ---
        top_layout = QVBoxLayout()
        top_layout.setSpacing(8)

        # Slider section (bit depth)
        slider_layout = QHBoxLayout()
        slider_layout.setSpacing(10)

        self.bit_label = QLabel("Bit Depth: 8-bit (256 colors)")
        self.bit_label.setAlignment(Qt.AlignCenter)

        self.bit_slider = QSlider(Qt.Horizontal)
        self.bit_slider.setMinimum(1)
        self.bit_slider.setMaximum(8)
        self.bit_slider.setValue(8)
        self.bit_slider.setTickPosition(QSlider.TicksBelow)
        self.bit_slider.setTickInterval(1)
        self.bit_slider.valueChanged.connect(self.update_bit_depth)

        slider_layout.addWidget(self.bit_label)
        slider_layout.addWidget(self.bit_slider)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)

        self.load_button = QPushButton("Load Image")
        self.save_button = QPushButton("Save Grayscale")
        self.load_button.clicked.connect(self.load_image)
        self.save_button.clicked.connect(self.save_grayscale)

        button_layout.addStretch(1)
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.save_button)
        button_layout.addStretch(1)

        # Combine controls
        top_layout.addLayout(slider_layout)
        top_layout.addLayout(button_layout)

        # --- Image display area ---
        image_layout = QHBoxLayout()
        image_layout.setSpacing(15)

        self.image_label = QLabel("No Image Loaded")
        self.gray_label = QLabel("No Image Loaded")

        for lbl in (self.image_label, self.gray_label):
            lbl.setStyleSheet("background-color: #E8E8E8; border: 1px solid #CCC;")
            lbl.setScaledContents(True)
            lbl.setAlignment(Qt.AlignCenter)

        image_layout.addWidget(self.image_label)
        image_layout.addWidget(self.gray_label)

        # Add everything
        main_layout.addLayout(top_layout)
        main_layout.addLayout(image_layout)
        self.setLayout(main_layout)

        # --- State ---
        self.original_image = None
        self.gray_image = None
        self.file_path = None
        self.bits = 8

        # Enable drag & drop
        self.setAcceptDrops(True)

    # --- Drag & Drop ---
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            self.load_image(file_path)

    # --- Load Image ---
    def load_image(self, file_path=None):
        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)"
            )
        if not file_path:
            return

        img = QImage(file_path)
        if img.isNull():
            QMessageBox.warning(self, "Error", "Failed to load image.")
            return

        self.file_path = file_path
        self.original_image = img
        self.update_grayscale()

    # --- Update Bit Depth ---
    def update_bit_depth(self, value):
        self.bits = value
        self.bit_label.setText(f"Bit Depth: {value}-bit ({2 ** value} colors)")
        if self.original_image:
            self.update_grayscale()

    # --- Grayscale Conversion (Fixed Padding Issue) ---
    def update_grayscale(self):
        img = self.original_image
        if img is None:
            return

        width = img.width()
        height = img.height()
        bytes_per_line = img.bytesPerLine()

        ptr = img.bits()
        ptr.setsize(img.byteCount())

        # Read raw buffer including padding
        arr = np.frombuffer(ptr, np.uint8).reshape((height, bytes_per_line))

        # Determine if RGB or RGBA, and strip padding correctly
        channels = 3
        if img.format() in (QImage.Format_RGB32, QImage.Format_ARGB32, QImage.Format_ARGB32_Premultiplied):
            channels = 4

        # Remove padding safely
        arr = arr[:, :width * channels]
        arr = arr.reshape((height, width, channels))

        # Drop alpha if exists
        if channels == 4:
            arr = arr[..., :3]

        # Grayscale conversion
        gray = np.dot(arr[..., :3], [0.299, 0.587, 0.114])

        # Bit-depth quantization
        levels = 2 ** self.bits
        gray_quant = np.round(gray / 255 * (levels - 1)) * (255 / (levels - 1))
        gray_quant = gray_quant.astype(np.uint8)

        # Convert to QImage
        gray_img = QImage(
            gray_quant.data,
            gray_quant.shape[1],
            gray_quant.shape[0],
            gray_quant.strides[0],
            QImage.Format_Grayscale8
        )

        # Display
        self.image_label.setPixmap(QPixmap.fromImage(img))
        self.gray_label.setPixmap(QPixmap.fromImage(gray_img))
        self.gray_image = gray_img
        self.adjust_image_labels()

    # --- Resize Handling ---
    def adjust_image_labels(self):
        width = self.width() // 2 - 30
        for lbl in [self.image_label, self.gray_label]:
            lbl.setFixedWidth(width)
            lbl.setScaledContents(True)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.adjust_image_labels()

    # --- Save Image ---
    def save_grayscale(self):
        if not self.gray_image:
            QMessageBox.warning(self, "Error", "No grayscale image to save.")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Grayscale Image", "grayscale.png",
            "PNG Files (*.png);;JPG Files (*.jpg);;BMP Files (*.bmp)"
        )
        if save_path:
            self.gray_image.save(save_path)
            QMessageBox.information(self, "Saved", f"Image saved successfully as:\n{save_path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageConverter()
    window.show()
    sys.exit(app.exec_())
