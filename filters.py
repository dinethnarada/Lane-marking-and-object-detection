import os
import cv2
import numpy as np
from utils import select_sub_mat,wrap

# Function - Normal Mean Filter
# Params - Wrapped image, image height, image width, kernel size
# Return - Filtered image array
# Description - Calculate mean of each kernel and apply
def mean_filter(wrapped_img, n, m, kernel_NXN):
    output_img = []
    for iter_row in range(n-kernel_NXN+1):
        output_row = []
        for iter_col in range(m-kernel_NXN+1):
            sub_mat = select_sub_mat(
                wrapped_img, iter_row, iter_col, kernel_NXN)
            sub_arr = [px for sub_row in sub_mat for px in sub_row]
            mean = sum(sub_arr)/(kernel_NXN*kernel_NXN)
            output_row.append(mean)
        output_img.append(output_row)
    return np.array(output_img)

# Function - Median Filter
# Params - Wrapped image, image height, image width, kernel size
# Return - Filtered image array
# Description - Sort each kernel values and apply median value
def median_filter(wrapped_img, n, m, kernel_NXN):
    output_img = []
    for iter_row in range(n-kernel_NXN+1):
        output_row = []
        for iter_col in range(m-kernel_NXN+1):
            sub_mat = select_sub_mat(
                wrapped_img, iter_row, iter_col, kernel_NXN)
            sub_arr = [px for sub_row in sub_mat for px in sub_row]
            sub_arr.sort()
            if(kernel_NXN % 2 == 0):
                median = (sub_arr[kernel_NXN*kernel_NXN//2] +
                          sub_arr[kernel_NXN*kernel_NXN//2-1])/2
            else:
                median = sub_arr[kernel_NXN*kernel_NXN//2]
            output_row.append(median)
        output_img.append(output_row)
    return np.array(output_img)


# Function - Apply mean, median and mid point filters to given image
#            according to user input
# Params - Image, Filter type(mean/median/mid), Kenel Size(int)
# Return - Filtered image array
def filter_func(input_img, filter_type, kernel_NXN=3):
    wrap_size = (kernel_NXN-1)//2
    n, m = len(input_img), len(input_img[0])

    # Wrap the image
    wrapped_img = wrap(input_img,n,m,wrap_size)

    n = len(wrapped_img)
    m = len(wrapped_img[0])
    if filter_type == 'mean':
        return mean_filter(wrapped_img, n, m, kernel_NXN) 
    elif filter_type == 'median':
        return median_filter(wrapped_img, n, m, kernel_NXN)
    elif filter_type == 'mid':
        return mid_point_filter(wrapped_img, n, m, kernel_NXN)