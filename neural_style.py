from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras import backend as K
from scipy.optimize import fmin_l_bfgs_b # Limited-memory Broyden-Fletcher-Goldfarb-Shanno
from typing import Dict, Text, List, Any
import numpy as np
import cv2
import os


class NeuralStyle:
    def __init__(
        self,
        settings: Dict[Text, Any]
    ) -> None:
        self.S = settings

        (w,h) = load_img(self.S['input_path']).size()
        self.dim = (h, w)

        # Tải ảnh nội dung và ảnh phong cách.
        # Ép kích thước của chúng về cùng một loại

        self.content = self.preprocess(self.S['imput_path'])
        self.style = self.preprocess(self.S['style_path'])

        # Khởi tạo một biến Kera từ đầu nội dung và ảnh phong cách
        self.content = K.variable(self.content)
        self.style = K.variable(self.style)

        # Cấp phát bộ nhớ cho ảnh đầu ra. 
        # Kích thước sẽ là [1,chiều cao, chiều rộng, 3]
        self.output = K.placeholder(
            shape=(
            1,           # Một mảng có để chứa ảnh
            self.dim[0], # Chiều cao
            self.dim[1], # Chiều rộng
            3            # 3 kênh màu
            )
        )

        # Ghép các tensor đầu vào thành 1 tensor duy nhất và đưa vào đầu vào.
        self.input = K.concatenate(
            tensors= [self.content, self.style, self.output],
            axis = 0
        )

        # Tải trọng số tiền huấn luyện.
        print("[INFO] loading network...")
        self.model = self.S["net"](
            weights="imagenet",
            include_top = False,
            input_tensor = self.input
        )

        # Xây dựng từ điển ánh xạ tên của mỗi lớp với trọng số bên trong. Để mình sẽ gom nhóm
        # Hay bỏ các lớp theo mong muốn.
        layer_map = {l.name: l.output for l in self.model.layers}








