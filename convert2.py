import numpy as np
from  matplotlib import pyplot as plt
from skimage.transform import resize
import tensorflow as tf
import PIL
import tensorflow.keras.applications.vgg19 as vgg19_model
from tensorflow.keras import Model


# # Get Images

# In[197]:
def load_image(image_path, resize=True):
	img = tf.io.read_file(image_path)
	img = tf.image.decode_image(img, channels=3)
	img = tf.image.convert_image_dtype(img, tf.float32)

	if resize:
		max_dim = 1280
		shape = tf.cast(tf.shape(img)[:-1], tf.float32)
		long_dim = max(shape)
		scale = max_dim / long_dim
		new_shape = tf.cast(shape * scale, tf.int32)
		img = tf.image.resize(img, new_shape)
		
	img = img[tf.newaxis, :]
	return img


'''Return a tf.model which can be called to get all intermediate layers from given list of layer names'''

def vgg_layers(total_layers):
	vgg = tf.keras.applications.VGG19(pooling = 'avg',include_top=False, weights='imagenet') #Create VGG Object
	vgg.trainable = False
	layers_list = [vgg.get_layer(name).output for name in total_layers] #Get all layers of given layer names
	feat_extraction_model = tf.keras.Model([vgg.input], outputs = layers_list)	#Create Object given input layers

	return feat_extraction_model


# In[199]:

'''Required for calculating style loss'''

def calc_gram_matrix(layer):
	
	result = tf.linalg.einsum('bijc,bijd->bcd', layer, layer)
	input_shape = tf.shape(layer) #get various features
	h_and_w = input_shape[1] * input_shape[2]
	 
	num_locations = tf.cast(h_and_w, tf.float32)
	return result/(num_locations)


# In[203]:


'''Function for creating layers'''

def create_layers(image):
	image = image*255
	processed = tf.keras.applications.vgg19.preprocess_input(image)
				   
	style_layers = style_layers_model(processed)
	style_outputs = [calc_gram_matrix(layer)for layer in style_layers]
	
	content_outputs = [content_layers_model(processed)]

	return [style_outputs,content_outputs]


'''given 'current' and 'target' layers, calculate total loss using MSE'''

def calc_loss(layers, outputs):
	total_loss = [] #total loss array is 1xn matrix, where n=number of content layers
	
	for layer in range(len(layers)):
		layer_loss = ((outputs[layer] - layers[layer])**2)
		total_loss.append(layer_loss)
	return total_loss


def tensor_to_image(tensor):
	tensor = tensor*255
	tensor = np.array(tensor, dtype=np.uint8)
	if np.ndim(tensor)>3:
		assert tensor.shape[0] == 1
		tensor = tensor[0]
	return PIL.Image.fromarray(tensor), tensor


'''Compute weighted loss function for both style and content'''

def style_content_loss(outputs):
	style_outputs = outputs[0]	 #current style
	content_outputs = outputs[1] #current content
	
	#List of loss at each layer
	style_loss = calc_loss(style_outputs, target_style_layers) 
	content_loss = calc_loss(content_outputs, target_content_layers)

	#sum of loss at every loss 
	style_loss_to_num = tf.add_n([tf.reduce_mean(layer) for layer in style_loss])
	content_loss_to_num = tf.add_n([tf.reduce_mean(layer) for layer in content_loss])
	
	#weighted sum
	style_loss_to_num *= (alpha / len(style_outputs)) #normalize
	content_loss_to_num *= (beta / len(content_outputs)) #normalize
	loss = style_loss_to_num + content_loss_to_num
	
	return loss


# In[220]:


#2. how does adam optimizer compare with LBFGS?



# In[221]:


'''Stochastic gradient descent to minimize image loss '''

@tf.function()
def gradient(image):
	
	#tf.compat.v1.enable_eager_execution()

	with tf.GradientTape() as tape: 

		#tape.watch(image) #output_image is variable we are differentiating on
		
		target_style_layers = [calc_gram_matrix(layer) for layer in style_layers_model(style_image)]
		target_content_layers = content_layers_model(content_image)
		
		outputs = create_layers(image)
		loss = style_content_loss(outputs) #our loss function
		loss += total_variation_weight*tf.image.total_variation(image)
		
	grad = tape.gradient(loss, image)
	opt.apply_gradients([(grad, image)])
	image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))


def train(epochs=10, steps=100, style_weight=1e-2, content_weight=1e4, variation_weight=30):
	global alpha, beta, total_variation_weight

	alpha = style_weight
	beta = content_weight
	total_variation_weight = variation_weight

	curr_step = 0
	for i in range(epochs):
		print('Training epoch:', i)
		for j in range(steps):
			curr_step += 1
			gradient(image)
	print("Done")



opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)	 

# Call this for every frame
def process_image(content_path, style_path):
	global style_image, content_image, content_layers_model, style_layers_model, target_style_layers, target_content_layers, image
	
	style_image = load_image(style_path)
	content_image = load_image(content_path)

	content_layers_names = ['block5_conv4']
	style_layers_names = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']

	content_layers_model = vgg_layers(content_layers_names) #content layers obj
	style_layers_model = vgg_layers(style_layers_names) #style layers obj

	target_style_layers = create_layers(style_image)[0]
	target_content_layers = create_layers(content_image)[1]

	## with their gram calc
	image = tf.Variable(content_image) #change to variable obj to pass into gradient
	
	train(epochs=4)
	
	return tensor_to_image(image)
	
	#ai = NeuralStyleTransfer(content_path, style_path)
	#content_layers_names = ['block5_conv4']
	#style_layers_names = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']
	#content_layers = ['block5_conv2']
	#style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
	#return ai.transfer(content_layers_names, style_layers_names, save=False, steps=100, epochs=4)
	

#####################################################################################################################################################################

import sys
import os

if __name__ == "__main__":
	src_dir = sys.argv[1]
	dst_dir = sys.argv[2]
	start_frame_id = int(sys.argv[3])
	end_frame_id = int(sys.argv[4])
	
	print('src:', src_dir)
	print('dst:', dst_dir)
	print('start:', start_frame_id)
	print('end:', end_frame_id)

	if not os.path.exists(dst_dir):
		os.makedirs(dst_dir)

	files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))] 
	files.sort(key = lambda x: int(x[5:-4]))

	for i in range(start_frame_id, min(len(files), end_frame_id)):
		file = os.path.join(src_dir, files[i])
		print('Processing', i, file, flush=True)

		frame = process_image(file, '../../media/scream.jpg')[0]
		#frame = process_image(file, 'media/scream.jpg')[0]
		frame.save(os.path.join(dst_dir, files[i]))
