


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


def save_files(saved_bboxes, filepath):
    '''
    saves bboxes in csv file
    '''
    my_df = pd.DataFrame(saved_bboxes, columns=['Image', 'Left Edge', 'Right Edge', 'Height', 'Confidence', "Area", "CenterX", "CenterY", "Radius"])
    my_df.to_csv(filepath, index=False)

class MainImage(QWidget): 
    '''
    this is the displayed image component that is shown on the screen of the UI
    image in the form of numpy array can be used to update this display image
    '''
    def __init__(self, main_app): 
        '''
        initializes object
        main ui window is inputed in
        no output
        '''
        super(MainImage, self).__init__()
        self.main_app = main_app
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
            
    def mousePressEvent(self, mouse_event):
        pass

    def mouseMoveEvent(self, mouse_event):
        pass

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

        self.checkbox = QCheckBox("Show Contours")
        self.checkbox.setChecked(True)
        self.checkbox.toggled.connect(main_app.box)

        self.listWidget = QListWidget()
        self.file_select = QPushButton('Select File')
        self.file_select.clicked.connect(main_app.selector)

        self.slidelabel1 = QLabel('confidence threshold:25')
        self.slider1 = QSlider(Qt.Horizontal)
        self.slider1.setMinimum(0)
        self.slider1.setMaximum(100)
        self.slider1.setValue(25)
        self.slider1.setTickPosition(QSlider.NoTicks)
        self.slider1.valueChanged.connect(main_app.valuechanged)

        self.group1 = QGroupBox('Select Directory')
        layout1 = QVBoxLayout(self)
        layout1.addWidget(self.listWidget)
        layout1.addWidget(self.file_select)
        self.group1.setLayout(layout1)

        self.group2 = QGroupBox('Options')
        layout2 = QVBoxLayout(self)
        layout2.addWidget(self.checkbox)
        layout2.addWidget(self.slidelabel1)
        layout2.addWidget(self.slider1)
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
        self.title = 'Helium Bubble Detector'
        self.left = 20
        self.top = 20
        self.width = 900
        self.height = 570
        self.working_image_path = ''
        self.displayed = False
        self.show_boxes = True
        self.conf_threshold = 25
        model_path = "best.pt"
        self.model = YOLO(model_path)
        self.init_window()


    def init_window(self):
        '''
        initializes widgets on the window including the toolbar and the image
        '''
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.painter = MainImage(self)
        self.installEventFilter(self.painter)

        self.toolbox = MainTool(self)
        self.toolbox.setMaximumWidth(250)

        self.widget1 = QWidget(self)
        self.top_layout = QHBoxLayout(self)
        self.open_button = QPushButton('Open File Directory')
        self.open_button.clicked.connect(self.on_open_button)
        self.file_path = QLineEdit(self)
        self.file_path.setObjectName("file path")
        self.file_path.returnPressed.connect(self.on_pushButtonOK_clicked)
        self.top_layout.addWidget(self.file_path)
        self.top_layout.addWidget(self.open_button)
        self.widget1.setLayout(self.top_layout)
        self.widget1.setMaximumHeight(100)

        self.table = QTableWidget(self)
        self.table_header = ['Image','Left Edge','Right Edge','Height','Confidence','Area']
        a = self.frameGeometry().width()//12
        self.table_col_width = [a, a, a, a, a, a]
        self.table_row_height = self.frameGeometry().height()//20
        self.table.setColumnCount(len(self.table_header))
        self.table.horizontalHeader().hide()
        self.table.verticalHeader().hide()
        self.table.setFont(QFont("arial", 9))

        self.save_button = QPushButton('Save Image and Boxes')
        self.save_button.clicked.connect(self.save_csv)
        self.delete_button = QPushButton('Delete Box')
        self.delete_button.clicked.connect(self.delete_box)

        self.bottom_buttons = QWidget(self)
        self.button_layout = QVBoxLayout()
        self.button_layout.addWidget(self.delete_button)
        self.button_layout.addWidget(self.save_button)
        self.bottom_buttons.setLayout(self.button_layout)

        self.bottom_widget = QWidget(self)
        self.bottom_layout = QHBoxLayout()
        self.bottom_layout.addWidget(self.table)
        self.bottom_layout.addWidget(self.bottom_buttons)
        self.bottom_widget.setLayout(self.bottom_layout)
        self.bottom_widget.setMaximumHeight(self.frameGeometry().height()//3)

        self.widget2 = QWidget(self)
        self.main_dock = QVBoxLayout()
        self.main_dock.addWidget(self.widget1)
        self.main_dock.addWidget(self.painter)
        self.main_dock.addWidget(self.bottom_widget)
        self.widget2.setLayout(self.main_dock)
        
        
        mainwidget = QWidget(self)
        layout = QHBoxLayout()
        layout.addWidget(self.toolbox)
        layout.addWidget(self.widget2)
        mainwidget.setLayout(layout)
        self.setCentralWidget(mainwidget)
        self.func_mappingSignal()

    def close_app(self):
        '''
        closing window
        '''
        sys.exit()

    def on_pushButtonOK_clicked(self):
        '''
        changes image path when button is clicked
        '''
        self.working_image_path = self.file_path.text()
        self.file_path.setText(str(self.working_image_path))
        self.displayed = False
        self.selected = False
        csv_list = glob.glob(os.path.join(self.working_image_path, "*.fts"))
        self.toolbox.listWidget.clear()
        for csv in csv_list:
            csv_fn = os.path.basename(csv)
            listWidgetItem = QListWidgetItem(csv_fn)
            self.toolbox.listWidget.addItem(listWidgetItem)

    
    def load_table(self):
        '''
        loads in table (called whenever table is updated)
        '''
        table_header = ['Image','Left Edge','Right Edge','Height','Confidence', 'Area']
        self.table.setRowCount(len(self.saved_bboxes) + 1)
        for row in range(len(self.saved_bboxes)):
            self.table.setRowHeight(row, self.table_row_height)
            for col in range(len(table_header)):
                if col != 0:
                    self.table.setItem(row+1, col, QTableWidgetItem(str(round(float(self.saved_bboxes[row][col]), 3))))
                else:
                    self.table.setItem(row+1, col, QTableWidgetItem(self.saved_bboxes[row][col]))

    def func_mappingSignal(self):
        '''
        forwards table data to func_test function whenever table is clicked
        '''
        self.table.clicked.connect(self.func_test)

    def func_test(self, item):
        '''
        updates self.selected_box variable to the box being currently selected by the user
        '''
        self.selected_box = item.row()-1
        self.redraw()

    def save_csv(self):
        '''
        saves the bounding boxes in a csv file
        '''
        if self.displayed == True:
            if self.selected == True:
                base = os.path.splitext(self.img_dir)[0]
                save_files(self.saved_bboxes, os.path.join(self.output_file_loc, base+".csv"))
                cv2.imwrite(os.path.join(self.output_file_loc, base+".png"), self.redraw(r=True))

    def delete_box(self):
        '''
        deletes selected row of table when delete is pressed
        '''
        if self.displayed:
            if self.selected:
                if self.selected_box >= 0:
                    del self.polar_coords[self.selected_box]
                    del self.saved_bboxes[self.selected_box]
                    self.selected_box = -1
                self.load_table()
                self.redraw()

    def redraw(self, r=False):
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

                    mask_overlay = np.zeros((h, w, 3), dtype=np.uint8)
                    ind = 0
                    for y in range(0, h, stride):
                        for x in range(0, w, stride):

                            for r in self.saved[ind]:
                                if r.masks is None:
                                    continue

                                boxes_conf = r.boxes.conf.cpu().numpy()
                                masks = r.masks.data.cpu().numpy()

                                for conf, mask in zip(boxes_conf, masks):

                                    if conf < self.conf_threshold/100:
                                        continue

                                    mask = cv2.resize(mask, (tile_size, tile_size))
                                    mask = (mask > 0.5).astype(np.uint8)

                                    colored_mask = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                                    colored_mask[:, :, 2] = mask * 255

                                    y1, y2 = y, min(y + tile_size, h)
                                    x1, x2 = x, min(x + tile_size, w)
                                    overlay_crop = colored_mask[:y2 - y1, :x2 - x1]
                                    mask_overlay[y1:y2, x1:x2] = np.maximum(mask_overlay[y1:y2, x1:x2], overlay_crop)
                            ind += 1
                    self.annotated_image = cv2.addWeighted(self.image, 0.7, mask_overlay, 0.5, 0)
                    w, h = self.painter.set_image(numpy=True, numpy_img=self.annotated_image)
                    return
            w, h = self.painter.set_image(numpy=True, numpy_img=self.image)

    def box(self):
        '''
        shows/hides boxes
        '''

        self.show_boxes = not self.show_boxes
        self.redraw()

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

    def selector(self):
        '''
        displays the selected image file
        '''
        self.selected = False
        self.boxes = []
        self.selected_box = -1
        self.table.setRowCount(1)

        for col in range(len(self.table_header)):
            self.table.setColumnWidth(col, self.table_col_width[col])
            self.table.setItem(0, col, QTableWidgetItem(self.table_header[col]))
        try:
            image_path = self.file_path.text()
            self.output_file_loc = image_path
            if os.path.isfile(os.path.join(image_path, self.toolbox.listWidget.selectedItems()[0].text())):
                
                self.image = cv2.imread(os.path.join(image_path, self.toolbox.listWidget.selectedItems()[0].text()))
                self.displayed = True
                self.redraw()
        except:
            pass

    def detector_button(self):
        '''
        runs image through the model and annotates image
        '''
        self.toolbox.checkbox.setChecked(True)
        self.show_boxes = True
        self.saved = []
        if self.displayed == True:

            tile_size = 640 
            overlap = 0.2  

            h, w = self.image.shape[:2]
            stride = int(tile_size * (1 - overlap))

            mask_overlay = np.zeros((h, w, 3), dtype=np.uint8)

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
                        verbose=False
                    )
                    self.saved.append(results)
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

                            colored_mask = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                            colored_mask[:, :, 2] = mask * 255 

                            y1, y2 = y, min(y + tile_size, h)
                            x1, x2 = x, min(x + tile_size, w)
                            overlay_crop = colored_mask[:y2 - y1, :x2 - x1]
                            mask_overlay[y1:y2, x1:x2] = np.maximum(mask_overlay[y1:y2, x1:x2], overlay_crop)

            self.annotated_image = cv2.addWeighted(self.image, 0.7, mask_overlay, 0.5, 0)
            w, h = self.painter.set_image(numpy=True, numpy_img=self.annotated_image)

            #self.load_table()

            self.selected = True

    def valuechanged(self):
        '''
        updates bboxes based on changing confidence value
        '''
        self.toolbox.slidelabel1.setText('confidence threshold: ' + str(self.toolbox.slider1.value()))
        self.conf_threshold = self.toolbox.slider1.value()

        self.redraw()
        
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
