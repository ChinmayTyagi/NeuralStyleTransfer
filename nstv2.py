import numpy as np
from  matplotlib import pyplot as plt
from skimage.transform import resize
import tensorflow as tf
import PIL
import tensorflow.keras.applications.vgg19 as vgg19_model
from tensorflow.keras import Model


# # Get Images

# In[197]:
def load_image(image_path):
	max_dim = 512
	image = tf.io.read_file(image_path)
	image = tf.image.decode_image(image, channels=3)
	image = tf.image.convert_image_dtype(image, tf.float32)
	shape = tf.cast(tf.shape(image)[:-1], tf.float32)
	
	long_dim = max(shape)
	scale = max_dim / long_dim

	new_shape = tf.cast(shape * scale, tf.int32)

	image = tf.image.resize(image, new_shape)
	image = image[tf.newaxis, :]
	return image

style = 'oil_image.jpg'
content = 'landscape.jpg'


style_image = load_image(style)
content_image = load_image(content)

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


content_layers_names = ['block5_conv4']
style_layers_names = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']

#Vgg model obj has avg pooling, preset imagenet as training data, and doesn't require top layers.
vgg = vgg19_model.VGG19(pooling = 'avg', weights='imagenet',include_top=False)

content_layers_model = vgg_layers(content_layers_names) #content layers obj
style_layers_model = vgg_layers(style_layers_names) #style layers obj

target_style_layers = create_layers(style_image)[0]
target_content_layers = create_layers(content_image)[1]


'''given 'current' and 'target' layers, calculate total loss using MSE'''

def calc_loss(layers, outputs):
	total_loss = [] #total loss array is 1xn matrix, where n=number of content layers
	
	for layer in range(len(layers)):
		layer_loss = ((outputs[layer] - layers[layer])**2)
		total_loss.append(layer_loss)
	return total_loss


#Tensor to image
id = 200
def convert_to_image(tensor, title):
	global id
	
	tensor = tensor * 225	
	final = np.array(tensor, dtype = np.uint8)
	
	#plt.figure(figsize = (12,12))
	#plt.imshow(final[0])
	#plt.title(title, fontsize = 18)
	#plt.axis('off')
	#plt.show()
	
	id += 1
	file_name = str(id) + '.jpg'
	print('Saving:', file_name)
	PIL.Image.fromarray(final[0]).save(file_name)


from PIL import Image, ImageDraw, ImageFont
def save_image_overlay(tensor):
	global id

	tensor = tensor * 225
	final = np.array(tensor, dtype=np.uint8)

	# Draw overlay information
	result_image = Image.new('RGB', (final[0].shape[1], final[0].shape[0]))
	result_image.paste(Image.fromarray(final[0]))
	result_image_draw = ImageDraw.Draw(result_image, mode='RGBA')
	result_image_draw.text((0, 0), 'Style Weight    : ' + str(alpha), fill='white', font=ImageFont.truetype('arial.ttf', 18))
	result_image_draw.text((0, 20), 'Content Weight  : ' + str(beta), fill='white', font=ImageFont.truetype('arial.ttf', 18))
	result_image_draw.text((0, 40), 'Variation Weight: ' + str(total_variation_weight), fill='white', font=ImageFont.truetype('arial.ttf', 18))

	# Save image
	result_image.save('imgs/' + str(id) + '.jpg')
	id += 1

# In[219]:


'''Compute weighted loss function for both style and content'''

#1. How does changing alpha/beta affect image quality?
alpha = 1e-1  # Style weight
beta = 1e6  # Content weight
total_variation_weight = 5

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

opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)	 
output_image = tf.Variable(content_image) #change to variable obj to pass into gradient


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


## with their gram calc
image = tf.Variable(content_image) #change to variable obj to pass into gradient


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
	save_image_overlay(image)
	print("Done")

# STOPPED AT 2 4 9
for i in range(2, 7):
	for j in range(10):
		for k in range(10):
			print('Training', i, j, k)
			train(epochs=6, style_weight=(1 / 10**i), content_weight=(1 * 10**j), variation_weight=k*25)
