import tensorflow as tf
import pix2pix  # copied from TensorFlow Examples, https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py

# This is the mask branch network
# Modified based on Tensorflow Tutorial, Image Segmentation example
# https://www.tensorflow.org/tutorials/images/segmentation
pre_trained_mobilenet = tf.keras.applications.MobileNetV2(input_shape=[224, 224, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 112x112
    'block_3_expand_relu',   # 56x56
    'block_6_expand_relu',   # 28x28
    'block_13_expand_relu',  # 14x14
    'block_16_project',      # 7x7
]
layers = [pre_trained_mobilenet.get_layer(name).output for name in layer_names]

# Create the feature extraction model
mask_branch_down_stack = tf.keras.Model(inputs=pre_trained_mobilenet.input, outputs=layers)
mask_branch_down_stack.trainable = False

mask_branch_up_stack = [
    pix2pix.upsample(512, 3),  # 7x7 -> 14x14
    pix2pix.upsample(256, 3),  # 14x14 -> 28x28
    pix2pix.upsample(128, 3),  # 28x28 -> 56x56
    pix2pix.upsample(64, 3),   # 56x56 -> 112x112
]

def mask_net():
  inputs = tf.keras.layers.Input(shape=[224, 224, 3]) # source color image

  # Downstack
  skips = mask_branch_down_stack(inputs)
  x1 = skips[-1]
  skips = reversed(skips[:-1])

  # Upstack
  for up, skip in zip(mask_branch_up_stack, skips):
    x1 = tf.keras.layers.Concatenate()([up(x1), skip])

  # Last layer of the mask branch, per-pixel classification
  dense_classification = tf.keras.layers.Conv2DTranspose(filters=2, kernel_size=3, strides=2,  padding='same')  #112x112 -> 224x224
  x = dense_classification(x1)

  return tf.keras.Model(inputs=inputs, outputs=x)

mask_model = mask_net()
# mask_model.load_weights('Line_train2.h5') # mask branch trained with mobile net freezed
mask_model.summary()
mask_function = tf.keras.backend.function(mask_model.inputs, mask_model.outputs)

# This is the PF branch
mobilenet = tf.keras.applications.MobileNetV2(input_shape=[224, 224, 3], include_top=False)
mobilenet.layers.pop(0)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 112x112
    'block_3_expand_relu',   # 56x56
    'block_6_expand_relu',   # 28x28
    'block_13_expand_relu',  # 14x14
    'block_16_project',      # 7x7
]
layers = [mobilenet.get_layer(name).output for name in layer_names]

# Start with the pre-trained mobilenet, leave the weight trainable
pf_down_stack = tf.keras.Model(inputs=mobilenet.input, outputs=layers)
pf_down_stack.trainable = True

up_stack_pf = [
    pix2pix.upsample(512, 3),  # 7x7 -> 14x14
    pix2pix.upsample(256, 3),  # 14x14 -> 28x28
    pix2pix.upsample(128, 3),  # 28x28 -> 56x56
    pix2pix.upsample(64, 3),   # 56x56 -> 112x112
]

def masked_pf_net():
  inp1 = tf.keras.layers.Input(shape=[224, 224, 3]) # source color image
  inp2 = tf.keras.layers.Input(shape=[224, 224, 2]) # source grey image + reference grey image
  inp3 = tf.keras.layers.Input(shape=[224, 224, 2]) # PF ground truth

  mask_model.trainable = False  # mask branch should have been trained separately
  x_mask = mask_model(inp1)
  x_mask = tf.keras.backend.argmax(x_mask)
  x_mask = tf.keras.backend.cast(x_mask, 'float32')
  x_mask = tf.keras.layers.Reshape((224, 224, 1))(x_mask) # this is the court line mask


  x2 = tf.cast(inp2, tf.float32) / 255.
  x2 = tf.keras.layers.Conv2D(filters=3, kernel_size=(1,1))(x2) # adaptor, from 2-channel to 3-channels
  skips = pf_down_stack(x2)
  x_pf = skips[-1]
  skips = reversed(skips[:-1])
  for up, skip in zip(up_stack_pf, skips):
    x_pf = tf.keras.layers.Concatenate()([up(x_pf), skip])

  last_upsample = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same')  #112x112 -> 224x224
  x_pf = last_upsample(x_pf)

  # this layer outputs the PF
  x_pf = tf.keras.layers.Conv2D(2, (1,1), name='output_layer_pf', use_bias=True, activation='linear')(x_pf)

  # ideally, x_pf should be the same as the ground truth on the court lines
  x_pf = (x_pf - inp3) * x_mask

  return tf.keras.Model(inputs=[inp1, inp2, inp3], outputs=x_pf)

model = masked_pf_net()
model.load_weights('masked_pfnet_v8_24000.h5')
model.summary()
output_pf = model.get_layer('output_layer_pf').output
pf_function = tf.keras.backend.function(model.inputs, output_pf)

# inference using color_img1, reference_img2
# grey_img1 = cv2.cvtColor(color_img1, cv2.COLOR_BGR2GRAY)
# mask = mask_function(color_img1.reshape(1, 224, 224, 3))[0]
# mask = (np.argmax(mask, axis=-1)).reshape(224, 224)
# pf = pf_function([color_img1.reshape(1, 224, 224, 3),
#                   np.array(np.stack([grey_img1, reference_img2, axis=-1)).reshape(1, 224, 224, 2),
#                   np.zeros((1, 224, 224, 2))])
# mask_pf = pf[0] * mask.reshape(224, 224, 1)
