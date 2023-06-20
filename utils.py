import tensorflow as tf
from typing import Text


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