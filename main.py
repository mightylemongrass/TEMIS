


#################################################################################################
#
#  Contains all the code for UI
#  Includes all code necessary for running UI
#  Uses functions from utils.py
#
#################################################################################################

import cv2
import numpy as np
import math
import sys
import copy
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from ultralytics import YOLO
import os
import glob
import pandas as pd
import traceback


segment_color = (255, 50, 0)


class MainImage(QWidget): 
    '''
    this is the displayed image component that is shown on the screen of the UI
    image in the form of numpy array can be used to update this display image
    '''
    def __init__(self, main_app, secondary=False): 
        '''
        initializes object
        main ui window is inputed in
        no output
        '''
        super(MainImage, self).__init__()
        self.main_app = main_app
        self.secondary = secondary
        self.image_pixmap = QPixmap(640, 640)
        self.image_pixmap.fill(Qt.white)
        self.image_scale = 1.0
        self.setMinimumSize(500, 400)
        self.show()

    def set_image(self, numpy=False, numpy_img=None):
        '''
        changing the image
        '''
        if numpy == True:
            numpy_img = QImage(numpy_img, numpy_img.shape[1],\
                                numpy_img.shape[0], numpy_img.shape[1] * 3,QImage.Format_RGB888)
            self.image_pixmap = QPixmap.fromImage(numpy_img)
        self.update()
        return self.image_pixmap.width(), self.image_pixmap.height()

    def paintEvent(self, event):
        '''
        changing the dimensions of the image when window is resized
        '''
        painter = QPainter()
        painter.begin(self)
        if self.image_pixmap and self.image_pixmap.size().width() > 0:
            paint_w = float(self.size().width())
            paint_h = float(self.size().height())
            image_w = float(self.image_pixmap.size().width())
            image_h = float(self.image_pixmap.size().height())
            
            resized_w = paint_w
            resized_h = paint_w * image_h / image_w

            if resized_h > paint_h:
                resized_w = paint_h * image_w / image_h
                resized_h = paint_h
            self.image_scale = resized_w / image_w
            resized = self.image_pixmap.scaled(int(resized_w), int(resized_h))
        painter.drawPixmap(0, 0, resized)
        painter.end()

    def widget_to_image_coords(self, x_widget, y_widget):
        if self.image_pixmap is None:
            return None, None

        image_w = self.image_pixmap.width()
        image_h = self.image_pixmap.height()

        widget_w = self.width()
        widget_h = self.height()

        resized_w = widget_w
        resized_h = widget_w * image_h / image_w
        if resized_h > widget_h:
            resized_w = widget_h * image_w / image_h
            resized_h = widget_h

        scale = image_w / resized_w

        x_img = int(x_widget * scale)
        y_img = int(y_widget * scale)

        x_img = max(0, min(image_w - 1, x_img))
        y_img = max(0, min(image_h - 1, y_img))

        return x_img, y_img
            
    def mousePressEvent(self, mouse_event):
        if not self.secondary:
            x_img, y_img = self.widget_to_image_coords(mouse_event.x(), mouse_event.y())
            self.main_app.delete_segment(x_img, y_img)
            self.main_app.delete_partially(x_img, y_img)
            self.main_app.draw_segments(x_img, y_img)
            
    def mouseMoveEvent(self, mouse_event):
        if not self.secondary:
            x_img, y_img = self.widget_to_image_coords(mouse_event.x(), mouse_event.y())
            self.main_app.delete_segment(x_img, y_img)
            self.main_app.delete_partially(x_img, y_img)
            self.main_app.draw_segments(x_img, y_img)

    def mouseReleaseEvent(self, mouse_event):
        pass
        

class MainTool(QWidget):
    '''
    UI component of the toolbar located on the left side of the UI window
    includes all parts of the toolbar with all its widgets (i.e. buttons, sliders, etc.)
    when widgets are interacted with, returns data back
    '''
    
    def __init__(self, main_app):
        '''
        initalization of widgets on the toolrack
        '''
        super(MainTool, self).__init__()
        self.main_app = main_app
        self.resize(170, 200)
        self.intValidator = QIntValidator()

        self.detector = QPushButton('Detect Bubbles')
        self.detector.clicked.connect(self.main_app.detector_button)

        self.checkbox = QCheckBox("Show Segments")
        self.checkbox.setChecked(True)
        self.checkbox.toggled.connect(main_app.box)

        self.listWidget = QListWidget()
        self.file_select = QPushButton('Select File')
        self.file_select.clicked.connect(main_app.selector)

        self.slidelabel_conf = QLabel('Confidence Threshold: 25')
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(0)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(25)
        self.conf_slider.setTickPosition(QSlider.NoTicks)
        self.conf_slider.valueChanged.connect(main_app.conf_valuechanged)

        self.group1 = QGroupBox('Select Directory')
        layout1 = QVBoxLayout(self)
        layout1.addWidget(self.listWidget)
        layout1.addWidget(self.file_select)
        self.group1.setLayout(layout1)

        self.group2 = QGroupBox('Options')
        layout2 = QVBoxLayout(self)
        layout2.addWidget(self.checkbox)
        layout2.addWidget(self.slidelabel_conf)
        layout2.addWidget(self.conf_slider)
        layout2.addWidget(self.detector)
        self.group2.setLayout(layout2)
        
        layout = QVBoxLayout(self)
        layout.addWidget(self.group1)
        layout.addWidget(self.group2)

        self.setLayout(layout)
        self.show()

            
class MainApp(QMainWindow): 
    '''
    Entirety of the UI
    components of the UI such as the MainImage and MainTool are integrated into here
    when UI is interacted with, the data is returned back to the code
    UI can be updated through MainApp
    '''

    def __init__(self):
        '''
        initalizes all variables
        '''
        super(MainApp, self).__init__()
        self.setFocusPolicy(Qt.StrongFocus)

        
        self.title = 'Transmission Electron Microscopy Image Segmentor (TEMIS)'
        self.left = 20
        self.top = 20

        if os.name == "nt":
            self.width = 1800
            self.height = 1000
        else:
            self.width = 1200
            self.height = 600
        self.working_image_path = ''
        self.selected = False
        self.displayed = False
        self.can_change_conf = True
        self.show_boxes = True
        self.conf_threshold = 25
        model_path = "best.pt"
        self.model = YOLO(model_path)
        self.init_window()
        self.draw_toggle = False
        self.delete_toggle = False
        self.delete_partially_toggle = False
        self.save_path = ""
        self.brush_rad = 25

    def init_window(self):
        '''
        initializes widgets on the window including the toolbar and the image
        '''
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.painter = MainImage(self)
        self.reference_image = MainImage(self, secondary=True)
        self.installEventFilter(self.painter)

        self.toolbox = MainTool(self)
        self.toolbox.setMaximumWidth(900)

        self.widget1 = QWidget(self)
        self.top_layout = QVBoxLayout(self)
        self.top_row = QHBoxLayout()
        self.bottom_row = QHBoxLayout()

        self.open_button = QPushButton('Open Image Directory')
        self.open_button.clicked.connect(self.on_open_button)
        self.file_path = QLineEdit(self)
        self.file_path.setObjectName("file path")
        self.file_path.returnPressed.connect(self.on_open_return)
        
        self.draw_button = QPushButton("Draw New Segment")
        self.draw_button.clicked.connect(self.draw_segments_toggle_func)
        self.delete_button = QPushButton("Delete Segment")
        self.delete_button.clicked.connect(self.delete_segments_toggle_func)
        self.delete_partially_button = QPushButton("Delete Partially")
        self.delete_partially_button.clicked.connect(self.delete_partially_toggle_func)

        self.draw_sliders_layout = QVBoxLayout(self)
        self.slidelabel_draw = QLabel('Brush Size: 20')
        self.draw_slider = QSlider(Qt.Horizontal)
        self.draw_slider.setMinimum(5)
        self.draw_slider.setMaximum(100)
        self.draw_slider.setValue(20)
        self.draw_slider.setTickPosition(QSlider.NoTicks)
        self.draw_slider.valueChanged.connect(self.draw_slider_changed)
        self.draw_sliders_layout.addWidget(self.slidelabel_draw)
        self.draw_sliders_layout.addWidget(self.draw_slider)
        
        self.top_row.addWidget(self.file_path)
        self.top_row.addWidget(self.open_button)
        self.bottom_row.addWidget(self.draw_button)
        self.bottom_row.addWidget(self.delete_button)
        ########### self.bottom_row.addWidget(self.delete_partially_button) ### modified
        self.bottom_row.addWidget(self.slidelabel_draw)
        self.bottom_row.addWidget(self.draw_slider)

        self.top_layout.addLayout(self.top_row)
        self.top_layout.addLayout(self.bottom_row) 

        self.widget1.setLayout(self.top_layout)
        self.widget1.setMaximumHeight(100)

        self.save_button = QPushButton('Save Image and Boxes')
        self.save_button.clicked.connect(self.save_csv)
        self.open_save_directory_button = QPushButton('Open Save Directory')
        self.open_save_directory_button.clicked.connect(self.on_save_button)
        self.save_file_path_line_edit = QLineEdit(self)
        self.save_file_path_line_edit.setObjectName("file path")
        self.save_file_path_line_edit.returnPressed.connect(self.on_save_return)

        self.size_slider_widget = QWidget(self)
        self.size_sliders_layout = QHBoxLayout(self)
        self.size_slider_draw = QLabel('Scale: 1 Pixel = ')
        self.size_slider = QLineEdit(self)
        self.size_slider.setText("0.05")
        self.size_sliders_layout.addWidget(self.size_slider_draw)
        self.size_sliders_layout.addWidget(self.size_slider)
        self.size_slider_widget.setLayout(self.size_sliders_layout)

        self.bottom_widget = QWidget(self)
        self.bottom_layout = QHBoxLayout()
        self.bottom_layout.addWidget(self.save_file_path_line_edit)
        self.bottom_layout.addWidget(self.open_save_directory_button) 
        self.bottom_layout.addWidget(self.save_button)
        self.bottom_widget.setLayout(self.bottom_layout)
        self.bottom_widget.setMaximumHeight(self.frameGeometry().height()//3)

        self.images = QWidget(self)
        self.images_layout = QHBoxLayout()
        self.images_layout.addWidget(self.reference_image)
        self.images_layout.addWidget(self.painter)
        self.images.setLayout(self.images_layout)

        self.widget2 = QWidget(self)
        self.main_dock = QVBoxLayout()
        self.main_dock.addWidget(self.widget1)
        self.main_dock.addWidget(self.images)
        self.main_dock.addWidget(self.size_slider_widget)
        self.main_dock.addWidget(self.bottom_widget)
        self.widget2.setLayout(self.main_dock)
        
        
        mainwidget = QWidget(self)
        splitter = QSplitter(Qt.Horizontal)

        splitter.addWidget(self.toolbox)
        splitter.addWidget(self.widget2)

        splitter.setSizes([250, 1000])

        splitter.setChildrenCollapsible(False)

        self.toolbox.setMinimumWidth(150)
        self.widget2.setMinimumWidth(400)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        mainwidget.setLayout(layout)
        self.setCentralWidget(mainwidget)


        layout = QHBoxLayout()
        layout.addWidget(self.toolbox)
        layout.addWidget(self.widget2)
        mainwidget.setLayout(layout)
        self.setCentralWidget(mainwidget)

    def close_app(self):
        '''
        closing window
        '''
        sys.exit()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.draw_segments_toggle_func()

    def on_open_button(self):
        '''
        opens the selected file
        '''
        if len(self.working_image_path) == 0:
            image_dir = os.getcwd()
        else:
            image_dir = os.path.dirname(self.working_image_path)
        self.working_image_path = QFileDialog.getExistingDirectory(self, 'Open File', image_dir)
        self.file_path.setText(str(self.working_image_path))
        self.displayed = False
        self.selected = False


        if os.name == "nt":
            csv_list = (
                glob.glob(os.path.join(self.working_image_path, "*.jpg")) +
                glob.glob(os.path.join(self.working_image_path, "*.png"))
            )
        else:
            csv_list = (
                glob.glob(os.path.join(self.working_image_path, "*.jpg")) +
                glob.glob(os.path.join(self.working_image_path, "*.png")) +
                glob.glob(os.path.join(self.working_image_path, "*.JPG")) +
                glob.glob(os.path.join(self.working_image_path, "*.PNG"))
            )
            
        self.toolbox.listWidget.clear()
        for csv in csv_list:
            csv_fn = os.path.basename(csv)
            listWidgetItem = QListWidgetItem(csv_fn)
            self.toolbox.listWidget.addItem(listWidgetItem)

    def on_open_return(self):
        '''
        changes image path when button is clicked
        '''
        self.working_image_path = self.file_path.text()
        self.file_path.setText(str(self.working_image_path))
        self.displayed = False
        self.selected = False
        csv_list = [f for f in glob.glob(os.path.join(self.working_image_path, "*")) if f.lower().endswith((".jpg", ".png"))]
        self.toolbox.listWidget.clear()
        for csv in csv_list:
            csv_fn = os.path.basename(csv)
            listWidgetItem = QListWidgetItem(csv_fn)
            self.toolbox.listWidget.addItem(listWidgetItem)

    def on_save_button(self):
        self.save_path = QFileDialog.getExistingDirectory(self, 'Open File', os.getcwd())
        self.save_file_path_line_edit.setText(str(self.save_path))
    
    def on_save_return(self):
        self.save_path = self.save_file_path_line_edit.text() 
        self.save_file_path_line_edit.setText(str(self.save_path))

    def save_csv(self):
        '''
        saves images and saves segments in a csv file
        '''
        if self.displayed == True:
            if self.selected:
                fn_prefix = os.path.splitext(self.toolbox.listWidget.selectedItems()[0].text())[0]
                if not os.path.isdir(self.save_path):
                    print("invalid save directory") 
                else:
                    try:
                        cv2.imwrite(os.path.join(self.save_path, fn_prefix + "_segment.png"), cv2.cvtColor(self.annotated_image, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(os.path.join(self.save_path, fn_prefix + "_segment_only.png"), self.mask_overlay)

                        mask = np.uint8(self.mask_overlay > 0)

                        contours, _ = cv2.findContours(
                            mask,
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE
                        )

                        data = []

                        for cnt in contours:
                            area = cv2.contourArea(cnt)

                            if area <= 20:
                                continue

                            M = cv2.moments(cnt)

                            if M["m00"] != 0:
                                cX = M["m10"] / M["m00"]
                                cY = M["m01"] / M["m00"]
                            else:
                                cX, cY = 0, 0

                            data.append({
                                "area (pixels)": area,
                                "area (nm)": round(area * (float(self.size_slider.text()) ** 2), 3),
                                "center_x (pixels)": round(cX, 3),
                                "center_y (pixels)": round(cY, 3)
                            })

                        df = pd.DataFrame(data)

                        df.to_csv(
                            os.path.join(self.save_path, fn_prefix + "_segment_data.csv"),
                            index=False
                        )                        
                    except Exception as e:
                        traceback.print_exc()
                        print(e)

    def redraw_conf(self, r=False):
        '''
        reloading image when checkboxes are clicked, different image preset is used, or confidence value is changed
        '''
        if self.displayed == True:
            if self.selected == True:
                if self.show_boxes == True:
                    tile_size = 640 
                    overlap = 0.2 

                    h, w = self.image.shape[:2]
                    stride = int(tile_size * (1 - overlap))

                    self.mask_overlay = np.zeros((h, w), dtype=np.int32)
                    self.conf_overlay = np.zeros((h, w), dtype=np.int32)
                    for det in self.detections:

                        if det["conf"] < self.conf_threshold / 100:
                            continue

                        x1, y1, x2, y2 = det["bbox"]

                        region = self.mask_overlay[y1:y2, x1:x2]
                        conf_region = self.conf_overlay[y1:y2, x1:x2]

                        mask = det["mask"]
                        conf = det["conf"]

                        update_pixels = mask & (conf > conf_region)

                        region[update_pixels] = 255
                        conf_region[update_pixels] = conf
 
                    self.annotated_image = self.image.copy()
                    self.annotated_image[self.mask_overlay > 0] = segment_color
                    w, h = self.painter.set_image(numpy=True, numpy_img=self.annotated_image) 
                    return
            w, h = self.painter.set_image(numpy=True, numpy_img=self.image)


    def redraw_edit(self):
        if self.displayed == True:
            if self.selected == True:
                if self.show_boxes == True:
                    self.annotated_image = self.image.copy()
                    self.annotated_image[self.mask_overlay > 0] = segment_color
                    w, h = self.painter.set_image(numpy=True, numpy_img=self.annotated_image)
                    return
        w, h = self.painter.set_image(numpy=True, numpy_img=self.image)


    def box(self):
        '''
        shows/hides boxes
        '''
        
        self.show_boxes = not self.show_boxes
        self.redraw_edit()

    def draw_segments_toggle_func(self):
        if self.selected and self.displayed:
            self.can_change_conf = False
            self.draw_toggle = not self.draw_toggle
            font = self.draw_button.font()
            font.setBold(self.draw_toggle)
            self.draw_button.setFont(font)
            if self.delete_toggle:
                self.delete_toggle = not self.delete_toggle 
                font = self.delete_button.font()
                font.setBold(self.delete_toggle)
                self.delete_button.setFont(font)
            if self.delete_partially_toggle:
                self.delete_partially_toggle = not self.delete_partially_toggle 
                font = self.delete_partially_button.font()
                font.setBold(self.delete_partially_toggle)
                self.delete_partially_button.setFont(font)

            if self.draw_toggle:
                self.painter.setCursor(Qt.CrossCursor)
            else:
                self.painter.setCursor(Qt.ArrowCursor)
                

    def draw_segments(self, y, x):
        if self.selected and self.displayed and self.show_boxes and self.draw_toggle:
            cv2.circle(self.mask_overlay, center=(y, x), radius=self.brush_rad, color=(255, 255, 255), thickness=-1)
            self.redraw_edit()

    def delete_segments_toggle_func(self):
        if self.selected and self.displayed:
            self.can_change_conf = False
            self.delete_toggle = not self.delete_toggle 
            font = self.delete_button.font()
            font.setBold(self.delete_toggle)
            self.delete_button.setFont(font)
            if self.draw_toggle:
                self.draw_toggle = not self.draw_toggle 
                font = self.draw_button.font()
                font.setBold(self.draw_toggle)
                self.draw_button.setFont(font)
            if self.delete_partially_toggle:
                self.delete_partially_toggle = not self.delete_partially_toggle 
                font = self.delete_partially_button.font()
                font.setBold(self.delete_partially_toggle)
                self.delete_partially_button.setFont(font)

            if self.delete_toggle:
                self.painter.setCursor(Qt.CrossCursor)
            else:
                self.painter.setCursor(Qt.ArrowCursor)

    def delete_segment(self, x, y):
        if not (self.selected and self.displayed and
                self.show_boxes and self.delete_toggle):
            return

        if self.mask_overlay[y, x] == 0:
            return

        num_labels, labels = cv2.connectedComponents(
            np.uint8(self.mask_overlay > 0)
        )

        clicked_label = labels[y, x]

        if clicked_label > 0:
            self.mask_overlay[labels == clicked_label] = 0

        self.redraw_edit()

    def delete_partially_toggle_func(self):
        if self.selected and self.displayed:
            self.can_change_conf = False
            self.delete_partially_toggle = not self.delete_partially_toggle 
            font = self.delete_partially_button.font()
            font.setBold(self.delete_partially_toggle)
            self.delete_partially_button.setFont(font)
            if self.delete_toggle:
                self.delete_toggle = not self.delete_toggle 
                font = self.delete_button.font()
                font.setBold(self.delete_toggle)
                self.delete_button.setFont(font)
            if self.draw_toggle:
                self.draw_toggle = not self.draw_toggle 
                font = self.draw_button.font()
                font.setBold(self.draw_toggle)
                self.draw_button.setFont(font)

    def delete_partially(self, x, y):
        if self.selected and self.displayed and self.show_boxes and self.delete_partially_toggle:
            cv2.circle(self.mask_overlay, center=(y, x), radius=self.brush_rad, color=(0, 0, 0), thickness=-1)
            self.redraw_edit()

    def draw_slider_changed(self):
        self.slidelabel_draw.setText('Brush Size: ' + str(self.draw_slider.value()))
        self.brush_rad = self.draw_slider.value()

    def selector(self):
        '''
        displays the selected image file
        '''
        self.selected = False
        self.boxes = []
        self.selected_box = -1
        self.can_change_conf = True

        try:
            image_path = self.file_path.text()
            self.output_file_loc = image_path
            if os.path.isfile(os.path.join(image_path, self.toolbox.listWidget.selectedItems()[0].text())):
                
                self.image = cv2.imread(os.path.join(image_path, self.toolbox.listWidget.selectedItems()[0].text()))
                self.displayed = True
                w, h = self.reference_image.set_image(numpy=True, numpy_img=self.image)
                self.redraw_conf()
        except:
            pass

    def mask_iou(self, mask1, mask2):
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()

        if union == 0:
            return 0.0

        return intersection / union


    def detection_iou(self, det1, det2):

        ax1, ay1, ax2, ay2 = det1["bbox"]
        bx1, by1, bx2, by2 = det2["bbox"]

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        if ix1 >= ix2 or iy1 >= iy2:
            return 0.0

        h = iy2 - iy1
        w = ix2 - ix1

        a_crop = det1["mask"][
            iy1 - ay1:iy2 - ay1,
            ix1 - ax1:ix2 - ax1
        ]

        b_crop = det2["mask"][
            iy1 - by1:iy2 - by1,
            ix1 - bx1:ix2 - bx1
        ]

        return self.mask_iou(a_crop, b_crop)


    def mask_nms(self, detections, iou_thresh=0.5):

        detections = sorted(
            detections,
            key=lambda d: d["conf"],
            reverse=True
        )

        keep = []

        while detections:

            best = detections.pop(0)
            keep.append(best)

            remaining = []

            for det in detections:

                if self.detection_iou(best, det) < iou_thresh:
                    remaining.append(det)

            detections = remaining

        return keep

    def detector_button(self):
        '''
        runs image through the model and annotates image
        '''
        self.toolbox.checkbox.setChecked(True)
        self.show_boxes = True
        self.detections = []
        if self.displayed == True:

            tile_size = 640 
            overlap = 0.2  

            h, w = self.image.shape[:2]
            stride = int(tile_size * (1 - overlap))

            ind = 1

            self.mask_overlay = np.zeros((h, w), dtype=np.int32)
            self.conf_overlay = np.zeros((h, w), dtype=np.float32)

            for y in range(0, h, stride):
                for x in range(0, w, stride):
                    tile = self.image[y:y + tile_size, x:x + tile_size]
                    th, tw = tile.shape[:2]

                    if th < tile_size or tw < tile_size:
                        padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                        padded[:th, :tw] = tile
                        tile = padded

                    results = self.model.predict(
                        source=tile,
                        imgsz=tile_size,
                        show=False,
                        conf=0.35,
                        verbose=False, 
                        retina_masks=True
                    )
                    for r in results:
                        if r.masks is None:
                            continue

                        boxes_conf = r.boxes.conf.cpu().numpy()
                        masks = r.masks.data.cpu().numpy()


                        for conf, mask in zip(boxes_conf, masks):

                            if conf < self.conf_threshold/100:
                                continue

                            mask = cv2.resize(mask, (tile_size, tile_size))
                            mask = (mask > 0.5).astype(np.uint8)

                            margin = 2

                            ys, xs = np.where(mask)

                            if len(xs) == 0:
                                continue

                            if (xs.min() <= margin or ys.min() <= margin or xs.max() >= tile_size - margin - 1 or ys.max() >= tile_size - margin - 1):
                                continue

                            y1, y2 = y, min(y + tile_size, h)
                            x1, x2 = x, min(x + tile_size, w)

                            overlay_crop = mask[:y2 - y1, :x2 - x1].astype(bool)

                            self.detections.append({
                                "mask": overlay_crop,
                                "bbox": (x1, y1, x2, y2),
                                "conf": float(conf)
                            })

            self.detections = self.mask_nms(
                self.detections,
                iou_thresh=0.2
            )

            for det in self.detections:

                x1, y1, x2, y2 = det["bbox"]

                region = self.mask_overlay[y1:y2, x1:x2]
                conf_region = self.conf_overlay[y1:y2, x1:x2]

                mask = det["mask"]
                conf = det["conf"]

                update_pixels = mask & (conf > conf_region)

                region[update_pixels] = 255
                conf_region[update_pixels] = conf


            self.annotated_image = self.image.copy()
            self.annotated_image[self.mask_overlay > 0] = segment_color

            w, h = self.painter.set_image(numpy=True,numpy_img=self.annotated_image)

            self.selected = True


    def conf_valuechanged(self):
        '''
        updates bboxes based on changing confidence value
        '''
        if self.can_change_conf:
            self.toolbox.slidelabel_conf.setText('confidence threshold: ' + str(self.toolbox.conf_slider.value()))
            self.conf_threshold = self.toolbox.conf_slider.value()

            self.redraw_conf()
        
    def on_run_button(self):
        if len(self.working_image_path) == 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText('No image selected')
            msg.setWindowTitle('Error')
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return

    def on_view_mode(self, mode):
        print("hello")



app = QApplication(sys.argv)
window = MainApp()
window.show()
sys.exit(app.exec_())
