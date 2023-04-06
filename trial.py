import tensorflow as tf

# tf.compat.v1.disable_eager_execution()

# hello = tf.constant('Hello, TensorFlow!')

# sess = tf.compat.v1.Session()

# print(sess.run(hello))
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices()))
