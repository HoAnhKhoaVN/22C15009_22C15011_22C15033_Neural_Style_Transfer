# from pyimagesearch.nn.conv import NeuralStyle
from keras.applications.vgg16 import VGG16



# initialize the settings dictionary
SETTINGS = {
# initialize the path to the input (i.e., content) image,
# style image, and path to the output directory
"input_path": "content_image.jpg",
"style_path": "style_image.jpg",
"output_path": "output",
# define the CNN to be used for style transfer, along with
# the set of content layer and style layers, respectively
"net": VGG16,
"content_layer": "block4_conv2",
"style_layers": ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"],
# store the content, style, and total variation weights,
# respectively
"content_weight": 1.0,
"style_weight": 100.0,
"tv_weight": 10.0,

# number of iterations
"iterations": 50,
}



