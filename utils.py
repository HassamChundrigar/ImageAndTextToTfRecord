import cv2 as cv
import codecs
import tensorflow as tf
import numpy as np

# Load Character set
chars=''
with codecs.open('chars.txt',encoding='utf-8') as f:
    chars=f.read()

# Resize Image to the same aspect ratio
def _resize(img,height):
    ratio=img.shape[1]/img.shape[0]
    return cv.resize(img,(int(height*ratio),height))

#Load Image, Resize to 64 height, Convert to Grayscale
def load_image(addr):
    img = cv.imread(addr)
    img = _resize(img,64)
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    return img

#Load Text and Convert it to dense
def load_text(addr):
    text=''
    with codecs.open(addr,encoding='utf-8') as file:
        text = file.read()
    dense = np.array(text_to_dense(text))
    return dense

#Convert String Text to dense array
def text_to_dense(text):
    dense=[]
    for char in text:
        index = chars.find(char)+1
        if (index>=0):
            dense.append(index)
        else:
            dense.append(len(chars)+1)
    return dense

#Dense to corresponding text removing Unidentified Character
def dense_to_text(dense):
    text=''
    for num in dense:
        if (num < len(chars)+1 and num > 0):
            text+=chars[num-1]
    return text

#Parsing for Reading Tf Records
def parse(serialized):
    # Define a dict with the data-names and types we expect to
    # find in the TFRecords file.
    # It is a bit awkward that this needs to be specified again,
    # because it could have been written in the header of the
    # TFRecords file instead.
    features = \
        {
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string)
        }

    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.parse_single_example(serialized=serialized,
                                             features=features)

    # Get the image as raw bytes.
    image_raw = parsed_example['image']

    # Decode the raw bytes so it becomes a tensor with type.
    image = tf.decode_raw(image_raw, tf.uint8)
    image = tf.reshape(image, (64,tf.cast(tf.shape(image)[0]/64,tf.int32)))
    # The type is now uint8 but we need it to be float.
    image = tf.cast(image, tf.float32)
    
    # Get the label associated with the image.
    label = parsed_example['label']
    label = tf.decode_raw(label, tf.uint8)
    # The image and label are now correct TensorFlow types.
    return image, label