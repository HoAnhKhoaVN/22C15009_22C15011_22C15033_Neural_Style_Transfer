import tensorflow as tf
from typing import Text
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False



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

def show_img(
    image : tf.Tensor,
    title: Text = None
)-> None:
    """_summary_

    Args:
        image (tf.Tensor): _description_
        title (Text, optional): _description_. Defaults to None.
    """
    if len(image.shape) > 3: # Có nhiều hơn 3 chiều
        image = tf.squeeze(input=image, axis= 0)

    assert len(image.shape) == 3, f"#Lỗi: Ảnh có {len(image.shape)} chiều!!!"

    plt.imshow(X = image)

    if title:
        plt.title(label= title)


if __name__ == "__main__":
    IMG_PATH = "content_image.jpg"
    content_image = load_img(path_to_img= IMG_PATH)
    show_img(
        image= content_image,
        title= "Content Image"
    )