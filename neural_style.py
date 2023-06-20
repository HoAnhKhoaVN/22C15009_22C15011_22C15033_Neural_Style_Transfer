from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras import backend as K
from scipy.optimize import fmin_l_bfgs_b # Limited-memory Broyden-Fletcher-Goldfarb-Shanno
from typing import Dict, Text, List, Any
import numpy as np
import cv2
import os
import tensorflow as tf



def load_img(
    path_to_img: Text,
)-> tf.Tensor:
    """_summary_

    Args:
        path_to_img (Text): _description_

    Returns:
        Tensor: _description_
    """
    max_dim = 512
    img = tf.io.read_file(filename = path_to_img)
    img = tf.image.decode_image(contents= img, channels=3, dtype= tf.float32)

    shape = tf.cast(tf.shape(input= img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    print(f"Shape : {shape} - Long_dim : {long_dim} - scale : {scale}")

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(
        images= img,
        size = new_shape
    )

    img = img[tf.newaxis, :]
    return img





    


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
        layer_map :Dict = {l.name: l.output for l in self.model.layers}

        # Trích xuất các đặc trưng tư các lớp nội dung
        content_features : tf.Tensor = layer_map[self.S['content_layer']]

        # Trích xuất các kích hoạt từ phong cách ảnh (chỉ mục 0)
        style_features : tf.Tensor = content_features[0, :, :, :] # Tại sao 0 là style ?????

        # Trích xuất các kích hoạt từ ảnh đầu ra (chỉ mục 2)
        output_features : tf.Tensor = content_features[2, :, :, :] # Tại sao chỉ mục 2 là nội dung.

        # Tính toán hàm lỗi cho các đặc trưng tái tạo nội dung
        content_loss = self.feauture_reconstruction(
            style_features,
            output_features
        )

        content_loss *= self.S['content_weight']

        # Tính lỗi phong cách
        # Khởi tạo biến lỗi phong cách và trọng số cho mỗi lớp phong cách được xét.
        style_loss = K.variable(value = 0.0)
        weight = 1.0 /len(self.S["style_layers"])

        # Lặp qua các lớp phong cách
        for layer in self.S['style_layers']:
            # Nắm bắt phong cách hiện tại và sử dụng chúng để trích xuất 
            # các đặc trưng về phong cách và đầu ra.
            style_output = layer_map[layer]
            style_features = style_output[1, :, :, :]
            output_features = style_features[2, :, :, :]

            # Tính toán lỗi "đạo nháy" phong cách
            T = self.style_recon_loss(
                style_features,
                output_features
            )
            style_loss += (weight * T)

        # Tính hàm lỗi phong cách
        style_loss *= self.S["style_weight"]
        total_variational_loss = self.S["tv_weight"] * self.total_variational_loss(self.output)
        total_loss = content_loss + style_loss + total_variational_loss

        # Tính toán đạo hàm của đầu ra và hàm lỗi
        grads = K.gradients(
            loss= total_loss,
            variables= self.output
        )

        outputs = [total_loss]
        outputs += grads

        # Tính độ lỗi và đạo hàm bằng phương pháp L-BFGS
        self.loss_and_grads = K.function(
            inputs= [self.output],
            outputs= outputs
        )



    def preprocess(
        self,

    )-> tf.Tensor:
        """_summary_

        Returns:
            Tensor: _description_
        """
        img = load_img()

    def feauture_reconstruction(
        self,

    )-> Tensor:
        """_summary_

        Returns:
            Tensor: _description_
        """
        pass

    def style_recon_loss(
        self,
    )-> Tensor:
        """_summary_

        Returns:
            Tensor: _description_
        """
        pass

    def total_variational_loss(
        self,
    )-> Tensor:
        """_summary_

        Returns:
            Tensor: _description_
        """
        pass







