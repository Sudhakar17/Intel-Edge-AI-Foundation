import cv2
import numpy as np


def handle_pose(output, input_shape):
    '''
    Handles the output of the Pose Estimation model.
    Returns ONLY the keypoint heatmaps, and not the Part Affinity Fields.
    
    The net outputs two blobs with shapes: [1, 38, 32, 57] and [1, 19, 32, 57]. 
    The first blob contains keypoint pairwise relations (part affinity fields), 
    the second one contains keypoint heatmaps.
    
    '''
    # TODO 1: Extract only the second blob output (keypoint heatmaps)
    # TODO 2: Resize the heatmap back to the size of the input
    
#     print(output.keys())
    
    keypoint_heatmaps = output['Mconv7_stage2_L2']
    resized_heatmaps = np.zeros((keypoint_heatmaps.shape[1], input_shape[0], input_shape[1]))
    for i in range(len(keypoint_heatmaps[0])):
        resized_heatmaps[i] = cv2.resize(keypoint_heatmaps[0][i],(input_shape[1], input_shape[0]))
    return resized_heatmaps


def handle_text(output, input_shape):
    '''
    Handles the output of the Text Detection model.
    Returns ONLY the text/no text classification of each pixel,
        and not the linkage between pixels and their neighbors.
    The net outputs two blobs. Refer to PixelLink and demos for details.
    [1x2x192x320] - logits related to text/no-text classification for each pixel.
    [1x16x192x320] - logits related to linkage between pixels and their neighbors.
    '''
    # TODO 1: Extract only the first blob output (text/no text classification)
    # TODO 2: Resize this output back to the size of the input
    
#     print(output.keys())
    
    logits = output['model/segm_logits/add']
    resized_logits = np.zeros((logits.shape[1], input_shape[0], input_shape[1]))
    for i in range(len(logits[0])):
        resized_logits[i] = cv2.resize(logits[0][i],(input_shape[1], input_shape[0]))

    return resized_logits


def handle_car(output, input_shape):
    '''
    Handles the output of the Car Metadata model.
    Returns two integers: the argmax of each softmax output.
    The first is for color, and the second for type.
    
    name: "color", shape: [1, 7, 1, 1] - Softmax output across seven color classes
    [white,gray, yellow, red, green, blue, black]
    name: "type", shape: [1, 4, 1, 1] - Softmax output across four type classes 
    [car, bus,  truck, van]
    
    '''
    # TODO 1: Get the argmax of the "color" output
    # TODO 2: Get the argmax of the "type" output
    
    color_idx = np.argmax(output['color'].flatten())
    type_idx =  np.argmax(output['type'].flatten())
    
    return color_idx,type_idx


def handle_output(model_type):
    '''
    Returns the related function to handle an output,
        based on the model_type being used.
    '''
    if model_type == "POSE":
        return handle_pose
    elif model_type == "TEXT":
        return handle_text
    elif model_type == "CAR_META":
        return handle_car
    else:
        return None


'''
The below function is carried over from the previous exercise.
You just need to call it appropriately in `app.py` to preprocess
the input image.
'''
def preprocessing(input_image, height, width):
    '''
    Given an input image, height and width:
    - Resize to width and height
    - Transpose the final "channel" dimension to be first
    - Reshape the image to add a "batch" of 1 at the start 
    '''
    image = np.copy(input_image)
    image = cv2.resize(image, (width, height))
    image = image.transpose((2,0,1))
    image = image.reshape(1, 3, height, width)

    return image