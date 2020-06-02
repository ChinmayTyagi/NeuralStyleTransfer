#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


import IPython.display as display

import numpy as np
import PIL.Image
import time
import functools


# In[3]:


def tensor_to_image(tensor):
	tensor = tensor*255
	tensor = np.array(tensor, dtype=np.uint8)
	if np.ndim(tensor)>3:
		assert tensor.shape[0] == 1
		tensor = tensor[0]
	return PIL.Image.fromarray(tensor)


# In[4]:


def load_img(path_to_img):
	max_dim = 512
	img = tf.io.read_file(path_to_img)
	img = tf.image.decode_image(img, channels=3)
	img = tf.image.convert_image_dtype(img, tf.float32)

	shape = tf.cast(tf.shape(img)[:-1], tf.float32)
	long_dim = max(shape)
	scale = max_dim / long_dim

	new_shape = tf.cast(shape * scale, tf.int32)

	img = tf.image.resize(img, new_shape)
	img = img[tf.newaxis, :]
	return img


# Content image
content_image = load_img('frame0.jpg')
style_image = load_img('media/apples.jpg')

def vgg_layers(layer_names):
	""" Creates a vgg model that returns a list of intermediate output values."""
	# Load our model. Load pretrained VGG, trained on imagenet data
	vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
	vgg.trainable = False

	outputs = [vgg.get_layer(name).output for name in layer_names]

	model = tf.keras.Model([vgg.input], outputs)
	return model


# In[11]:


def gram_matrix(input_tensor):
	result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
	input_shape = tf.shape(input_tensor)
	num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
	return result/(num_locations)


# In[13]:


class StyleContentModel(tf.keras.models.Model):
	def __init__(self, style_layers, content_layers):
		super(StyleContentModel, self).__init__()
		self.vgg =	vgg_layers(style_layers + content_layers)
		self.style_layers = style_layers
		self.content_layers = content_layers
		self.num_style_layers = len(style_layers)
		self.vgg.trainable = False

	def call(self, inputs):
		"Expects float input in [0,1]"
		inputs = inputs*255.0
		preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
		outputs = self.vgg(preprocessed_input)
		style_outputs, content_outputs = (outputs[:self.num_style_layers], 
										  outputs[self.num_style_layers:])

		style_outputs = [gram_matrix(style_output)
						 for style_output in style_outputs]

		content_dict = {content_name:value 
						for content_name, value 
						in zip(self.content_layers, content_outputs)}

		style_dict = {style_name:value
					  for style_name, value
					  in zip(self.style_layers, style_outputs)}

		return {'content':content_dict, 'style':style_dict}


image = tf.Variable(content_image)


# In[17]:


def clip_0_1(image):
	return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


# In[18]:


opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)


def style_content_loss(outputs):
	style_outputs = outputs['style']
	content_outputs = outputs['content']
	style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
						   for name in style_outputs.keys()])
	style_loss *= style_weight / num_style_layers

	content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
							 for name in content_outputs.keys()])
	content_loss *= content_weight / num_content_layers
	loss = style_loss + content_loss
	return loss


# In[21]:


@tf.function()
def train_step(image):
	with tf.GradientTape() as tape:
		outputs = extractor(image)
		loss = style_content_loss(outputs)
		loss += total_variation_weight*tf.image.total_variation(image)

	grad = tape.gradient(loss, image)
	opt.apply_gradients([(grad, image)])
	image.assign(clip_0_1(image))


# Don't touch these, they are global variables that are set by the train function below
num_content_layers = None
num_style_layers = None
extractor = None
style_targets = None
content_targets = None

def train(content_layers, style_layers, epochs=10, steps_per_epoch=100):
	global num_content_layers, num_style_layers, extractor, style_targets, content_targets
	
	num_content_layers = len(content_layers)
	num_style_layers = len(style_layers)
	extractor = StyleContentModel(style_layers, content_layers)
	style_targets = extractor(style_image)['style']
	content_targets = extractor(content_image)['content']
	
	# Start training
	import time
	start = time.time()
	print('Training...')

	step = 0
	for n in range(epochs):
		for m in range(steps_per_epoch):
			step += 1
			train_step(image)
		print("Train step: {}".format(step))

	end = time.time()
	print("Processed in {:.1f} seconds".format(end-start))


style_weight=1e-2
content_weight=1e4
total_variation_weight=30

train(['block5_conv2'], ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'], epochs=4, steps_per_epoch=100)
tensor_to_image(image).save('output-cw_' + str(content_weight) + '-sw_' + str(style_weight) + '-vw_' + str(total_variation_weight) + '.jpg')

