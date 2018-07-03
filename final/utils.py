import numpy as np
import skimage.io as imageio

"""
readImgList
    - input : a list of image filenames
    - output: a 4D numpy array (N, H, W, C)

transformLabel
    - input : original id labels
    - output: re-defined labels by y_dict

one_hot_encoding
    - input : a list of classes
    - output: a 2D numpy array (N, num_classes) 
"""
def readImgList(file_list):
    images = list()
    for i, file in enumerate(file_list):
        print(i, end="\r")
        img = imageio.imread(file)
        img = img.astype(int)
        images.append(img)
    return np.array(images)

def transformLabel(id_list, y_dict):
    label = list()
    for uid in list(id_list):
        label.append(y_dict[uid])
    return np.array(label)

def one_hot_encoding(class_numbers, num_classes):
    return np.eye(num_classes, dtype=float)[class_numbers]

"""
count_number_params
    - 

get_params_shape

count_flops

countFlopsParas

"""

def count_number_params(para_dict):
    n = 0
    for k,v in sorted(para_dict.items()):
        if 'bn_mean' in k:
            continue
        elif 'bn_variance' in k:
            continue
        elif 'gamma' in k:
            continue
        elif 'beta' in k:
            continue
        elif 'conv' in k or 'fc' in k:
            n += get_params_shape(v[0].shape.as_list())
            n += get_params_shape(v[1].shape.as_list())
    return n

def get_params_shape(shape):
    n = 1
    for dim in shape:
        n = n*dim
    return n

def count_flops(para_dict, net_shape, input_shape=(3, 218, 178)):
    # Format:(channels, rows,cols)
    total_flops_per_layer = 0
    input_count = 0
    for k,v in sorted(para_dict.items()):
        if 'bn_mean' in k:
            continue
        elif 'bn_variance' in k:
            continue
        elif 'gamma' in k:
            continue
        elif 'beta' in k:
            continue
        elif 'fc' in k:
            continue
        elif 'conv' in k:
            conv_filter = v[0].shape.as_list()[3::-1] # (64 ,3 ,3 ,3)  # Format: (num_filters, channels, rows, cols)
            stride = 1
            padding = 1

            if conv_filter[1] == 0:
                n = conv_filter[2] * conv_filter[3] # vector_length
            else:
                n = conv_filter[1] * conv_filter[2] * conv_filter[3]  # vector_length

            flops_per_instance = n + ( n -1)    # general defination for number of flops (n: multiplications and n-1: additions)

            num_instances_per_filter = (( input_shape[1] - conv_filter[2] + 2 * padding) / stride) + 1  # for rows
            num_instances_per_filter *= ((input_shape[2] - conv_filter[3] + 2 * padding) / stride) + 1  # multiplying with cols

            flops_per_filter = num_instances_per_filter * flops_per_instance
            total_flops_per_layer += flops_per_filter * conv_filter[0]  # multiply with number of filters

            total_flops_per_layer += conv_filter[0] * input_shape[1] * input_shape[2] # bias

            input_shape = net_shape[input_count].as_list()[3:0:-1]
            input_count +=1

    total_flops_per_layer += net_shape[-1].as_list()[1] *2360*2
    return total_flops_per_layer

def countFlopsParas(net, input_shape):
    total_flops = count_flops(net.para_dict, net.net_shape, input_shape=input_shape)
    if total_flops / 1e9 > 1:   # for Giga Flops
        print(total_flops/ 1e9 ,'{}'.format('GFlops'))
    else:
        print(total_flops / 1e6 ,'{}'.format('MFlops'))

    total_params = count_number_params(net.para_dict)

    if total_params / 1e9 > 1:   # for Giga Flops
        print(total_params/ 1e9 ,'{}'.format('G'))
    else:
        print(total_params / 1e6 ,'{}'.format('M'))
    
    return total_flops, total_params