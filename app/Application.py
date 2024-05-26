import sys

from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights, \
    ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from pathlib import Path
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QGridLayout, QFileDialog, \
    QPushButton, QLineEdit, QCheckBox


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.weights = None
        self.model = None

        self.initialize_model()

        self.is_object_detection = True
        self.image_path = ""
        self.labels = ""

        self.setWindowTitle("Image detector")
        self.setMinimumSize(QSize(1000, 600))

        self.filename_edit = QLineEdit()

        self.image_label = QLabel()
        self.image_label.setPixmap(QPixmap())
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(500)

        self.categorials_label = QLabel(self.labels)

        file_browse_button = QPushButton('Browse')
        file_browse_button.clicked.connect(self.open_file_dialog)

        object_detection_checkbox = QCheckBox()
        object_detection_checkbox.setCheckState(Qt.CheckState.Checked)
        object_detection_checkbox.stateChanged.connect(self.switch_object_detection)

        main_layout = QVBoxLayout()
        option_layout = QGridLayout()

        main_layout.addWidget(self.image_label)
        main_layout.addLayout(option_layout)

        option_layout.setVerticalSpacing(0)
        option_layout.addWidget(QLabel('Image categories:'), 0, 0)
        option_layout.addWidget(self.categorials_label, 0, 1)
        option_layout.addWidget(QLabel('Enable object detection:'), 1, 0)
        option_layout.addWidget(object_detection_checkbox, 1, 1)
        option_layout.addWidget(QLabel('File:'), 2, 0)
        option_layout.addWidget(self.filename_edit, 2, 1)
        option_layout.addWidget(file_browse_button, 2, 2)

        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

    def open_file_dialog(self):
        filename, ok = QFileDialog.getOpenFileName(
            self,
            "Select a File",
            "./",
            "Images (*.png *.jpg)"
        )
        if filename:
            path = Path(filename)
            self.image_path = str(path)
            self.filename_edit.setText(str(path))
        self.update_interface()

    def update_interface(self):
        if self.is_object_detection:
            self.detection()
            path = "./images/processed_image.jpg"
        else:
            path = self.image_path
            self.labels = ""
        self.image_label.setPixmap(QPixmap(path))
        self.categorials_label.setText(self.labels)

    def switch_object_detection(self, s):
        self.is_object_detection = bool(s)
        self.update_interface()

    def initialize_model(self, model_name="SSDLite320"):
        if model_name == "FasterRCNN":
            self.weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            self.model = fasterrcnn_resnet50_fpn_v2(weights=self.weights, box_score_thresh=0.8)
        elif model_name == "SSDLite320":
            self.weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
            self.model = ssdlite320_mobilenet_v3_large(weights=self.weights, box_score_thresh=0.8)
        self.model.eval()

    def detection(self, box_threshold=0.8):
        img = read_image(self.image_path)

        preprocess = self.weights.transforms()

        batch = [preprocess(img)]

        prediction = self.model(batch)[0]
        n = sum(prediction["scores"] > box_threshold)
        prediction = {"boxes": prediction["boxes"][0:n], "scores": prediction["scores"][0:n],
                      "labels": prediction["labels"][0:n]}
        labels = [self.weights.meta["categories"][i] for i in prediction["labels"]]
        self.labels = ", ".join((i for i in labels))
        box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                                  labels=labels,
                                  colors="red",
                                  width=4)
        im = to_pil_image(box.detach())
        im.save("./images/processed_image.jpg")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    app.exec()
