import tensorflow as tf

# Tạo hai biến TensorFlow Variable
var1 = tf.Variable([1, 2, 3])
var2 = tf.Variable([4, 5, 6])

# Nối hai biến theo trục 0
concatenated = tf.concat([var1, var2], axis=0)

# In kết quả
print("Kết quả nối (concatenation):", concatenated.numpy())
print(f"Type concatenated: {type(concatenated)}")