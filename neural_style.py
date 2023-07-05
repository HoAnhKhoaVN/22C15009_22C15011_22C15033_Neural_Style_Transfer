from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras import backend as K
from scipy.optimize import fmin_l_bfgs_b # Limited-memory Broyden-Fletcher-Goldfarb-Shanno
from typing import Dict, Text, List, Any
import numpy as np
import cv2
import os
import tensorflow as tf
from PIL import Image










class NeuralStyle:
    def __init__(
        self,
        settings: Dict[Text, Any]
    ) -> None:
        self.S = settings

        (w,h) = load_img(self.S['input_path']).size
        self.dim = (h, w)

        # Tải ảnh nội dung và ảnh phong cách.
        # Ép kích thước của chúng về cùng một loại

        self.content = self.preprocess(self.S['input_path'])
        self.style = self.preprocess(self.S['style_path'])

        # Khởi tạo một biến Kera từ đầu nội dung và ảnh phong cách
        self.content = tf.Variable(
            self.content,
            name= "content"
            )
        self.style = tf.Variable(
            self.style,
            name= "style"
        )

        # Cấp phát bộ nhớ cho ảnh đầu ra.
        # Khởi tạo một biến với giá trị ngẫu nhiên từ phân phối chuẩn
        initializer = tf.random_normal_initializer(
            mean= 0,
            stddev= 0.05,
            seed=555
        )

        # Kích thước sẽ là [1,chiều cao, chiều rộng, 3]
        shape_output =(
            1,           # Một mảng có để chứa ảnh
            self.dim[0], # Chiều cao
            self.dim[1], # Chiều rộng
            3            # 3 kênh màu
            )


        self.output = tf.Variable(
            initial_value=initializer(shape=shape_output),
            name= "output"
        )

        # Ghép các tensor đầu vào thành 1 tensor duy nhất và đưa vào đầu vào.
        self.input = tf.concat(
            values= [self.content, self.style, self.output],
            axis = 0,
            name= "input"
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

        print(layer_map)

        # # Trích xuất các đặc trưng tư các lớp nội dung
        # content_features : tf.Tensor = layer_map[self.S['content_layer']]

        # # Trích xuất các kích hoạt từ phong cách ảnh (chỉ mục 0)
        # style_features : tf.Tensor = content_features[0, :, :, :] # Tại sao 0 là style ?????

        # # Trích xuất các kích hoạt từ ảnh đầu ra (chỉ mục 2)
        # output_features : tf.Tensor = content_features[2, :, :, :] # Tại sao chỉ mục 2 là nội dung.

        # # Tính toán hàm lỗi cho các đặc trưng tái tạo nội dung
        # content_loss = self.feauture_reconstruction(
        #     style_features,
        #     output_features
        # )

        # content_loss *= self.S['content_weight']

        # # Tính lỗi phong cách
        # # Khởi tạo biến lỗi phong cách và trọng số cho mỗi lớp phong cách được xét.
        # style_loss = K.variable(value = 0.0)
        # weight = 1.0 /len(self.S["style_layers"])

        # # Lặp qua các lớp phong cách
        # for layer in self.S['style_layers']:
        #     # Nắm bắt phong cách hiện tại và sử dụng chúng để trích xuất
        #     # các đặc trưng về phong cách và đầu ra.
        #     style_output = layer_map[layer]
        #     style_features = style_output[1, :, :, :]
        #     output_features = style_features[2, :, :, :]

        #     # Tính toán lỗi "đạo nháy" phong cách
        #     T = self.style_recon_loss(
        #         style_features,
        #         output_features
        #     )
        #     style_loss += (weight * T)

        # # Tính hàm lỗi phong cách
        # style_loss *= self.S["style_weight"]
        # total_variational_loss = self.S["tv_weight"] * self.total_variational_loss(self.output)
        # total_loss = content_loss + style_loss + total_variational_loss

        # # Tính toán đạo hàm của đầu ra và hàm lỗi
        # grads = K.gradients(
        #     loss= total_loss,
        #     variables= self.output
        # )

        # outputs = [total_loss]
        # outputs += grads

        # # Tính độ lỗi và đạo hàm bằng phương pháp L-BFGS
        # self.loss_and_grads = K.function(
        #     inputs= [self.output],
        #     outputs= outputs
        # )



    def preprocess(
        self,
        path: Text,
    )-> tf.Tensor:
        """_summary_

        Returns:
            Tensor: _description_
        """
        # Tải ảnh lên
        img : tf.Tensor = load_img(
            path = path,
            target_size = self.dim
        )

        # Chuyển đổi tensor thành NumPy array
        img :np.ndarray = img_to_array(img)

        # Thêm một chiều vào img
        img :np.ndarray = np.expand_dims(
            a = img,
            axis= 0
        )

        # Tiền xử lý ảnh theo kiểu VGG16
        # -> Kết quả trả về ảnh BRG
        img :np.ndarray = preprocess_input(img)

        # Xuất ảnh ra để xem thử
        save_img = Image.fromarray(
            obj=img.squeeze(),
            mode= "RGB"
        )

        save_img.save(
            fp= f"preprocess_{path.split('/')[-1].split('.')[0]}.png"
        )

        return img


    def feauture_reconstruction(
        self,

    )-> tf.Tensor:
        """_summary_

        Returns:
            Tensor: _description_
        """
        pass

    def style_recon_loss(
        self,
    )-> tf.Tensor:
        """_summary_

        Returns:
            Tensor: _description_
        """
        pass

    def total_variational_loss(
        self,
    )-> tf.Tensor:
        """_summary_

        Returns:
            Tensor: _description_
        """
        pass

if __name__ == "__main__":
    from stype_transfer import SETTINGS
    ns = NeuralStyle(
        settings= SETTINGS
    )

    # ns.preprocess(
    #     path= "style_image.jpg"
    # )
