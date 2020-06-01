import tensorflow as tf

import numpy as np
import PIL.Image
import time
import functools

def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def tensor_to_image(tensor):
        tensor = tensor*255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor)>3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return PIL.Image.fromarray(tensor), tensor

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


class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
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


class NeuralStyleTransfer:

    def __init__(self, content_image_path, style_image_path):
        self._content_image = load_img(content_image_path)
        self._style_image = load_img(style_image_path)
        self._image = tf.Variable(self._content_image)

    def style_content_loss(self, outputs):
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-self.style_targets[name])**2) 
                            for name in style_outputs.keys()])
        style_loss *= self.style_weight / self.num_style_layers

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-self.content_targets[name])**2) 
                                for name in content_outputs.keys()])
        content_loss *= self.content_weight / self.num_content_layers
        loss = style_loss + content_loss
        return loss

    @tf.function()
    def train_step(self, image):
        with tf.GradientTape() as tape:
            outputs = self.extractor(image)
            loss = self.style_content_loss(outputs)
            loss += self.total_variation_weight*tf.image.total_variation(image)

        grad = tape.gradient(loss, image)
        self.opt.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))

    def transfer(self, content_layers, style_layers, style_weight=1e-2, content_weight=1e4, total_variation_weight=30, epochs=10, steps=100, save=True):
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.total_variation_weight = total_variation_weight

        self.opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

        self.num_content_layers = len(content_layers)
        self.num_style_layers = len(style_layers)
        self.extractor = StyleContentModel(style_layers, content_layers)
        self.style_targets = self.extractor(self._style_image)['style']
        self.content_targets = self.extractor(self._content_image)['content']
        
        # Start training
        import time
        start = time.time()
        print('Training for', epochs * steps, 'steps...')

        step = 0
        for n in range(epochs):
            for m in range(steps):
                step += 1
                self.train_step(self._image)
            print("Train step: {}".format(step))

        end = time.time()
        print("Processed in {:.1f} seconds".format(end-start))

        result = tensor_to_image(self._image)
        if (save):
           result[0].save('output-cw_' + str(content_weight) + '-sw_' + str(style_weight) + '-vw_' + str(total_variation_weight) + '.jpg') 
        return result


# Call this for every frame
def process_image(content_path, style_path):
    ai = NeuralStyleTransfer(content_path, style_path)
    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    return ai.transfer(content_layers, style_layers, save=False, steps=20, epochs=1)
    

#####################################################################################################################################################################

import cv2 
import os 
import shutil
import numpy as np

class VideoInfo:
    def __init__(self, name, fps):
        self.name = name
        self.fps = fps

def get_path():
    while True:
        rel_path = input('Enter relative path of video to process: ')
        if os.path.exists(rel_path):
            return rel_path
        else:
            print("File does not exist, try again")
    
def video_to_frames(file_path):
    vid = cv2.VideoCapture(file_path)
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    try: 
        if not os.path.exists(f'output_frames'): 
            os.mkdir(f'output_frames') 
    except OSError: 
        print ('Error: Failed to create output_frames directory') 

    currentframe = 0

    ret, frame = vid.read() 
    while(ret): 
        name = f'./output_frames/frame{str(currentframe)}.jpg'
        print(f'Creating..{name}')

        cv2.imwrite(name, frame) 
        currentframe += 1
        ret, frame = vid.read()

    fps = 0
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver)  < 3 :
        fps = vid.get(cv2.cv.CV_CAP_PROP_FPS)
        print(f"Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {fps}")
    else :
        fps = vid.get(cv2.CAP_PROP_FPS)
        print(f"Frames per second using video.get(cv2.CAP_PROP_FPS) : {fps}")

    vid.release() 
    cv2.destroyAllWindows() 
    return VideoInfo(file_name, fps)


def frames_to_video(video_info):
    dir_path = 'output_frames/'
    files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))] 
    files.sort(key = lambda x: int(x[5:-4]))

    #frames = []
    size = ()

    for i in range(len(files)):
        print('Processing frame', i)
        filename = dir_path + files[i]
        
        
        frame = process_image(filename, 'media/apples.jpg')[0]
        frame.save(filename)
        
        
        
        frame = cv2.imread(filename)
        h, w, layers = frame.shape
        size = (w, h)
        #frames.append(frame)
        
        if i == 100:
            break
    
    out = cv2.VideoWriter(f'{video_info.name}_stylized.mp4',cv2.VideoWriter_fourcc(*'mp4v'), video_info.fps, size)

    print(f"Writing final video to '{video_info.name}_stylized.mp4'...")
    for f in files:
        frame = cv2.imread(f)
        out.write(frame)
    out.release()

def clear_working_dirs():
    print("Clearing directory 'output_frames'... ")
    if os.path.exists(f'output_frames'): 
        shutil.rmtree('output_frames')
        
    
#####################################################################################################################################################################


if __name__ == "__main__":
    clear_working_dirs()
    video_info = video_to_frames(get_path())
    frames_to_video(video_info)

    
    

