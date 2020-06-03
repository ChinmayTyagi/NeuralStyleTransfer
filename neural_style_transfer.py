
import numpy as np
from  matplotlib import pyplot as plt
from skimage.transform import resize
import tensorflow as tf
import PIL
import keras.applications.vgg19 as vgg19_model
from keras import Model


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

'''Return a tf.model which can be called to get all intermediate layers from given list of layer names'''
def vgg_layers(total_layers):
    vgg = tf.keras.applications.VGG19(pooling = 'avg',include_top=False, weights='imagenet') #Create VGG Object
    vgg.trainable = False
    layers_list = [vgg.get_layer(name).output for name in total_layers] #Get all layers of given layer names
    feat_extraction_model = tf.keras.Model([vgg.input], outputs = layers_list)  #Create Object given input layers

    return feat_extraction_model

'''given 'current' and 'target' layers, calculate total loss using MSE'''
def calc_loss(layers, outputs):
    total_loss = [] #total loss array is 1xn matrix, where n=number of content layers
    
    for layer in range(len(layers)):
        layer_loss = ((outputs[layer] - layers[layer])**2)
        total_loss.append(layer_loss)
    return total_loss

def calc_gram_matrix(layer):
    
    result = tf.linalg.einsum('bijc,bijd->bcd', layer, layer)
    input_shape = tf.shape(layer) #get various features
    h_and_w = input_shape[1] * input_shape[2]
     
    num_locations = tf.cast(h_and_w, tf.float32)
    return result/(num_locations)

def create_layers(image):
    image = image*255
    processed = tf.keras.applications.vgg19.preprocess_input(image)
                   
    style_layers_obj = style_layers_model(processed)
    style_outputs = [calc_gram_matrix(layer)for layer in style_layers_obj]
    
    content_outputs = [content_layers_model(processed)]

    return [style_outputs,content_outputs]

'''Tensor to image'''
def convert_to_image(tensor, title):  
    tensor = tensor * 225   
    final = np.array(tensor, dtype = np.uint8)
    plt.figure(figsize = (12,12))
    plt.imshow(final[0])
    plt.title(title, fontsize = 18)
    plt.axis('off')
    plt.show()
    return PIL.Image.fromarray(final[0])


'''Compute weighted loss function for both style and content'''
alpha = 1e-2
beta = 1e4
total_variation_weight=30

def style_content_loss(outputs):
    style_outputs = outputs[0]   #current style
    content_outputs = outputs[1] #current content
    
    #List of loss at each layer
    style_loss = calc_loss(style_outputs, style_targets) 
    content_loss = calc_loss(content_outputs, content_targets)

    #sum of loss at every loss 
    style_loss_to_num = tf.add_n([tf.reduce_mean(layer) for layer in style_loss])
    content_loss_to_num = tf.add_n([tf.reduce_mean(layer) for layer in content_loss])
    
    #weighted sum
    style_loss_to_num *= (alpha / len(style_outputs)) #normalize
    content_loss_to_num *= (beta / len(content_outputs)) #normalize
    loss = style_loss_to_num + content_loss_to_num
    
    return loss

'''Stochastic gradient descent to minimize image loss '''
@tf.function()
def gradient(image):
    
    with tf.GradientTape() as tape: 
        
        outputs = create_layers(image)
        loss = style_content_loss(outputs) #our loss function
        loss += total_variation_weight*tf.image.total_variation(image)
        
    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))


#######################################################################
#Testing
'''Global Vars'''

style = '/Users/chinmay/Desktop/175/oil_image.jpg'
content = '/Users/chinmay/Desktop/175/landscape.jpg'

style_image = load_image(style)
content_image = load_image(content)

content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

#similar to extractor
vgg = vgg19_model.VGG19(pooling = 'avg', weights='imagenet',include_top=False)
content_layers_model = vgg_layers(content_layers) #content layers obj
style_layers_model = vgg_layers(style_layers) #style layers obj
style_targets = create_layers(style_image)[0]
content_targets = create_layers(content_image)[1]
# Inputs
opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)  
output_image = tf.Variable(content_image) #change to variable obj to pass into gradient
def train(epochs=1, steps_per_epoch=10):

    # Start training
    import time
    start = time.time()
    print('Training...')

    step = 0
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            gradient(output_image)
        print("Train step: {}".format(step))

    end = time.time()
    print("Processed in {:.1f} seconds".format(end-start))
    return step

num_step = train(epochs=1, steps_per_epoch=1)
folder_path = '/Users/chinmay/Desktop/NeuralStyleTransfer-master/stills/Output_Iteration' + str(num_step)
final_im = convert_to_image(output_image, str("After " + str(num_step) + " Iterations"))
final_im.save(folder_path, 'JPEG')

