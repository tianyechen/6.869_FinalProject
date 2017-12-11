from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import cv2, numpy as np
from matplotlib import pyplot as plt
from keras import backend as K
import imageio
import scipy.ndimage
import scipy as sp

base_model = VGG16(weights='imagenet')
model1 = Model(inputs=base_model.input, outputs=base_model.get_layer('block1_conv2').output)
model2 = Model(inputs=base_model.input, outputs=base_model.get_layer('block2_conv2').output)
model3 = Model(inputs=base_model.input, outputs=base_model.get_layer('block3_conv3').output)
model4 = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_conv3').output)

img_path = 'liang_bw.bmp'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

def extract_hypercolumns(imgs=[], layers=[], base_model=VGG16(weights='imagenet')):
	hypercolumns = []
	models = []
	for layer in layers:
		models.append(Model(input=base_model.input, outputs=base_model.get_layer(layer).output))
	for img in imgs:
		img_hypercolumns = []
		for i in range(len(layers)):
			model = models[i]
			features = model.predict(img)
			view_features = features.transpose((0,3,1,2))
			avg_features = np.sum(view_features[0], 0) / len(view_features[0])
			upscaled = sp.misc.imresize(avg_features, size=(224, 224), mode="F", interp='bilinear')
			img_hypercolumns.append(upscaled)
		img_avg = np.sum(img_hypercolumns, 0) / len(img_hypercolumns[0])
		hypercolumns.append(img_avg)
	return np.asarray(hypercolumns)

hypercolumns = extract_hypercolumns([x], ['block1_conv2'])

for i in range(len(hypercolumns)):
	hypercolumn = (hypercolumns[i] * 255 / np.max(hypercolumns[i])).astype('uint8')
	f = plt.figure()
	plt.imshow(hypercolumn)
	imageio.imwrite('img_'+str(i)+'_features.bmp', hypercolumn)

plt.show()

# for hypercolumn in hypercolumns:
# 	f = plt.figure()
# 	plt.imshow(hypercolumn)

# f = plt.figure()
# avg_hypercolumns = np.sum(hypercolumns, 0) / len(hypercolumns[0])
# plt.imshow(avg_hypercolumns)
# plt.show()