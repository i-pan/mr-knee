"""
Helper functions.
"""
import pydicom
import numpy as np

def channels_last_to_first(img):
    """ Move the channels to the first dimension."""
    img = np.swapaxes(img, 0,2)
    img = np.swapaxes(img, 1,2)
    return img 

def preprocess_input(img, model): 
    """ Preprocess an input image. """
    # assume image is RGB 
    img = img[..., ::-1].astype('float32')
    model_min = model.input_range[0] ; model_max = model.input_range[1] 
    img_min = float(np.min(img)) ; img_max = float(np.max(img))
    img_range = img_max - img_min 
    model_range = model_max - model_min 
    if img_range == 0: img_range = 1. 
    img = (((img - img_min) * model_range) / img_range) + model_min 
    img[..., 0] -= model.mean[0] 
    img[..., 1] -= model.mean[1] 
    img[..., 2] -= model.mean[2] 
    img[..., 0] /= model.std[0] 
    img[..., 1] /= model.std[1] 
    img[..., 2] /= model.std[2] 
    return img

def to_categorical(y, num_classes=None):
    """
    Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    Arguments:
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    Returns:
        A binary matrix representation of the input. The classes axis is placed
        last.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
      input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
      num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def get_image_from_dicom(dicom_file): 
    """
    Extract the image as an array from a DICOM file.
    """
    dcm = pydicom.read_file(dicom_file) 
    array = dcm.pixel_array 
    try:
        array *= int(dcm.RescaleSlope)
        array += int(dcm.RescaleIntercept)
    except:
        pass 
    if dcm.PhotometricInterpretation == "MONOCHROME1": 
        array = np.invert(array.astype("uint16")) 
    array = array.astype("float32") 
    array -= np.min(array) 
    array /= np.max(array) 
    array *= 255. 
    return array.astype('uint8')

class LossTracker(): 
    #
    def __init__(self, num_moving_average=1000): 
        self.losses = []
        self.loss_history = []
        self.num_moving_average = num_moving_average
    #
    def update_loss(self, minibatch_loss): 
        self.losses.append(minibatch_loss) 
    # 
    def get_avg_loss(self): 
        self.loss_history.append(np.mean(self.losses[-self.num_moving_average:]))
        return self.loss_history[-1]
    # 
    def reset_loss(self): 
        self.losses = [] 
    # 
    def get_loss_history(self): 
        return self.loss_history


