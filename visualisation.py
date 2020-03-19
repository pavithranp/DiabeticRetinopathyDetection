
import tensorflow as tf
import pathlib
import tensorflow.keras as k
import numpy as np
from tensorflow.python.framework.ops import disable_eager_execution

import matplotlib.pyplot as plt
import cv2
root_path = "../Retinopathy"
#root_path = "/home/pavithran/datasets/Retinopathy" #change according to retinopathy folder
train_directory = root_path + "/OriginalImages/Training"
test_directory = root_path + "/OriginalImages/Testing"

batch_size = 10
epochs = 100
learning_rate = 0.001
inputs = k.layers.Input(shape=(256,256,3))

# a layer instance is callable on a tensor, and returns a tensor

conv2d1 = k.layers.Conv2D(32, (3, 3),strides=(1, 1), activation='relu')(inputs)
#maxpool1 = k.layers.MaxPooling2D(pool_size=(2, 2))(conv2d1)
#Dropouta = k.layers.Dropout(0.3)(maxpool1)
BN1 = k.layers.BatchNormalization()(conv2d1)
LLR1 = k.layers.LeakyReLU()(BN1)

conv2d2 = k.layers.Conv2D(32, (3, 3),strides=(2, 2), activation='relu')(LLR1)
#maxpool2 = k.layers.MaxPooling2D(pool_size=(2, 2))(conv2d2)
#Dropoutb = k.layers.Dropout(0.3)(maxpool2)
BN2 = k.layers.BatchNormalization()(conv2d2)
LLR2 = k.layers.LeakyReLU()(BN2)

conv2d2a = k.layers.Conv2D(64, (3, 3),strides=(2, 2), activation='relu')(LLR2)
#maxpool2a = k.layers.MaxPooling2D(pool_size=(2, 2))(conv2d2a)
#Dropoutc = k.layers.Dropout(0.3)(conv2d2a)
BN2a = k.layers.BatchNormalization()(conv2d2a)
LLR2a = k.layers.LeakyReLU()(BN2a)

conv2d2b = k.layers.Conv2D(128, (3, 3),strides=(2, 2), activation='relu')(LLR2a)
#maxpool2a = k.layers.MaxPooling2D(pool_size=(2, 2))(conv2d2a)
#Dropoutc = k.layers.Dropout(0.3)(conv2d2a)
BN2b = k.layers.BatchNormalization()(conv2d2b)
LLR2b = k.layers.LeakyReLU()(BN2b)

GAP = k.layers.GlobalAveragePooling2D()(LLR2b)

flatten = k.layers.Flatten()(GAP)

output = k.layers.Dense(2, activation='softmax')(flatten)
model = k.models.Model(inputs=inputs, outputs=output)

opt = k.optimizers.RMSprop(learning_rate)

model.compile(optimizer=opt ,loss="sparse_categorical_crossentropy" , metrics= ['accuracy'])

checkpoint_path = root_path + '/training_1/weights-improvement-106-0.70.hdf5'
latest = tf.train.latest_checkpoint(checkpoint_path)
print(latest)
print(checkpoint_path)
model.load_weights(checkpoint_path)
model.summary()

# visualisation

fileread = train_directory+'/IDRiD_002.jpg'
print(fileread)
image_string = tf.io.read_file(fileread) # Load image with path "fn"

img = cv2.imread(fileread)[...,::-1]
img = cv2.resize(img, (256, 256))
image = tf.expand_dims(tf.cast(img, tf.float32)/255.0, axis=0)
x = model.predict(image)
grad_model = tf.keras.models.Model([model.inputs], [model.get_layer("conv2d_3").output, model.output])
filter_index = 0
#
class_output = model.output[:,1]
with tf.GradientTape() as tape:

    conv_outputs, predictions = grad_model(image)
    loss = predictions[:, 1]

output = conv_outputs[0]
print(output.shape)
print(loss)
grads = tape.gradient(loss, conv_outputs)[0]
print(grads.shape)
gate_f = tf.cast(output > 0, 'float32')
gate_r = tf.cast(grads > 0, 'float32')
guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads


weights = np.mean(guided_grads, axis=(0, 1))

cam = np.ones(output.shape[0:2], dtype=np.float32)

for index, w in enumerate(weights):
    cam += w * output[:, :, index]


# Heatmap visualization
#print(image_decoded.shape)

cam = cv2.resize(cam.numpy(), (256, 256))
cam = np.maximum(cam, 0)

heatmap = (cam - cam.min()) / (cam.max() - cam.min())

cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)

output_image = cv2.addWeighted(img, 1, cam, 0.5, 0) # place with weights of cam here(4th argument)


fig=plt.figure(figsize=(15, 45))

fig.add_subplot(1, 3, 1)
plt.title("Actual Image")
plt.imshow(img)
fig.add_subplot(1, 3, 2)
plt.title("Grad Cam features")
plt.imshow(cam)

fig.add_subplot(1, 3, 3)
plt.title("gradcam overlayed on Actual Image")
plt.imshow(output_image)

plt.show()