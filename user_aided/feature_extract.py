# from keras.applications.vgg16 import VGG16

# VGG16(include_top=True, weights='imagenet',
#           input_tensor=None, input_shape=None,
#           pooling=None,
#           classes=1000):

# model = VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import cv2, numpy as np
from matplotlib import pyplot as plt
from keras import backend as K

model = VGG16(weights='imagenet', include_top=False)

# img_path = 'tianye.jpg'
# img = image.load_img(img_path, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)

# features = model.predict(x)

# pic=features[0,:,:,100]
# plt.imshow(pic)
# plt.gray()
# plt.show()

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')
# out = model.predict(im)
# print np.argmax(out)

im_original = cv2.resize(cv2.imread('liang_bb.jpg'), (224, 224))
# im = im_original.transpose((2,0,1))
im = np.expand_dims(im_original, axis=0)
# print im.shape

im_converted = cv2.cvtColor(im_original, cv2.COLOR_BGR2RGB)
# plt.imshow(im_converted)
# plt.show()

# f1 = plt.figure()
# out = model.predict(im)
# plt.plot(out.ravel())
# plt.show()

# get_feature = K.function([model.layers[0].input], [model.layers[3].output])
# feat = get_feature([im, 0])
# print len(feat[0])
# print len(feat[0][0])
# print len(feat[0][0][0])
# plt.imshow(feat[0][0][0])
# plt.show()

base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block1_pool').output)

img_path = 'liang_bw.bmp'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)
print features.shape
view_features = features.transpose((0,3,1,2))
view_features = np.sum(view_features[0], 0) / 128
plt.imshow(view_features)
plt.show()
# print view_features.shape
# print view_features[0][0]
# print np.around(view_features[0][0], decimals=2)

# print len(features[0])
# print len(features[0][0])
# print len(features[0][0][0])
# print len(block4_pool_features[0][0][0][0])