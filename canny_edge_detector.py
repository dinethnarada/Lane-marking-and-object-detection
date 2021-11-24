import os
import numpy as np
import cv2
import math
from utils import select_sub_mat


def convolution(input_img, kernel):
    n = len(input_img)
    m = len(input_img[0])
    kernel_NXN = len(kernel)
    wrap_size = (kernel_NXN-1)//2

    if(isinstance(input_img, list) == False):
        input_img = input_img.tolist()
    wrapped_img = wrap(input_img, n, m, wrap_size)
    
    conv = []
    n += wrap_size*2
    m += wrap_size*2
    for i in range(n - kernel_NXN+1):
        conv_row = []
        for j in range(m - kernel_NXN+1):
            sub_mat = select_sub_mat(wrapped_img, i, j, kernel_NXN)
            sm = sum_mul(sub_mat, kernel, kernel_NXN)
            conv_row.append(sm)
        conv.append(conv_row)
    return conv


def sum_mul(sub_mat, kernel, kernel_NXN):
    sum_mul = 0
    for i in range(kernel_NXN):
        for j in range(kernel_NXN):
            sum_mul += sub_mat[i][j]*kernel[i][j]
    return sum_mul


def wrap(input_img, n, m, wrap_size):
    wrapped_img = []
    for r in range(n):
        wrapped_row = [input_img[r][-1]]*wrap_size + input_img[r]+[input_img[r][-1]]*wrap_size
        wrapped_img.append(wrapped_row)
    for iter in range(wrap_size):
        wrapped_img.insert(0, wrapped_img[-1])
        wrapped_img.append(wrapped_img[0])
    return wrapped_img


def mat_hypot(mat_x, mat_y):
    mat_x_n = len(mat_x)
    mat_x_m = len(mat_x[0])
    mat_y_n = len(mat_y)
    mat_y_m = len(mat_y[0])

    if(mat_x_n != mat_y_n or mat_x_m != mat_y_m):
        # error
        return -1
    mat = []
    for i in range(mat_x_n):
        mat_row = []
        for j in range(mat_x_m):
            mat_row.append(math.hypot(mat_x[i][j], mat_y[i][j]))
        mat.append(mat_row)
    return mat


def mat_arctan(mat_x, mat_y):
    mat_x_n = len(mat_x)
    mat_x_m = len(mat_x[0])
    mat_y_n = len(mat_y)
    mat_y_m = len(mat_y[0])

    if(mat_x_n != mat_y_n or mat_x_m != mat_y_m):
        # error
        return -1
    mat = []
    for i in range(mat_x_n):
        mat_row = []
        for j in range(mat_x_m):
            mat_row.append(math.atan2(mat_y[i][j], mat_x[i][j]))
        mat.append(mat_row)
    return mat


def gaussian_kernel(kernel_NXN, sigma):
    kernel = []
    for i in range(kernel_NXN):
        kernel_row = []
        for j in range(kernel_NXN):
            x = i-(kernel_NXN-1)//2
            y = j-(kernel_NXN-1)//2
            kernel_cell = math.exp(-1*((math.hypot(x, y)**2) /
                                   (2*sigma**2)))/(2*math.pi*sigma**2)
            kernel_row.append(kernel_cell)
        kernel.append(kernel_row)
    return kernel

# step 2:filter image with agaussian filter


def gaussian_filter(input_img, sigma, kernel_NXN):
    kernel = gaussian_kernel(kernel_NXN, sigma)
    filtered_img = convolution(input_img, kernel)
    return filtered_img

# step 3:estimate gradient strength and direction


def gradient(filtered_img):
    mx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    my = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

    gx = convolution(filtered_img, mx)
    gy = convolution(filtered_img, my)

    strength_mat = mat_hypot(gx, gy)
    direction_mat = mat_arctan(gx, gy)

    return (strength_mat, direction_mat)

# step 4:non maxima suppression


def non_maxima_suppression(strength_mat, direction_mat):
    n = len(strength_mat)
    m = len(strength_mat[0])
    suppressed_img = [[0 for _ in range(m)]for _ in range(n)]
    for i in range(1, n-1):
        for j in range(1, m-1):
            if((direction_mat[i][j] >= math.pi/8 and direction_mat[i][j] < 3*math.pi/8) or (direction_mat[i][j] >= -7*math.pi/8 and direction_mat[i][j] < -5*math.pi/8)):
                suppressed_img[i][j] = strength_mat[i][j] if strength_mat[i][j] == max(
                    strength_mat[i-1][j+1], strength_mat[i+1][j-1], strength_mat[i][j]) else 0
            elif((direction_mat[i][j] >= 3*math.pi/8 and direction_mat[i][j] < 5*math.pi/8) or (direction_mat[i][j] >= -5*math.pi/8 and direction_mat[i][j] < -3*math.pi/8)):
                suppressed_img[i][j] = strength_mat[i][j] if strength_mat[i][j] == max(
                    strength_mat[i-1][j], strength_mat[i+1][j], strength_mat[i][j]) else 0
            elif((direction_mat[i][j] >= 5*math.pi/8 and direction_mat[i][j] < 7*math.pi/8) or (direction_mat[i][j] >= -3*math.pi/8 and direction_mat[i][j] < -1*math.pi/8)):
                suppressed_img[i][j] = strength_mat[i][j] if strength_mat[i][j] == max(
                    strength_mat[i-1][j-1], strength_mat[i+1][j+1], strength_mat[i][j]) else 0
            else:
                suppressed_img[i][j] = strength_mat[i][j] if strength_mat[i][j] == max(
                    strength_mat[i][j-1], strength_mat[i][j+1], strength_mat[i][j]) else 0

    return suppressed_img


def max_mat(mat):
    return max([max(row) for row in mat])

# step 5:dual threshold


def dual_threshold(suppressed_img, high, low):
    high_threshold = max_mat(suppressed_img)*high
    low_threshold = high*low
    n = len(suppressed_img)
    m = len(suppressed_img[0])
    linked_mat = []
    # p[i][j]>=high ---> 255
    # p[i][j]<=low ---> 0
    # high>p[i][j]>low  ---> -1
    for i in range(n):
        linked_mat_row = []
        for j in range(m):
            if(suppressed_img[i][j] >= high_threshold):
                linked_mat_row.append(255)
            elif(suppressed_img[i][j] <= low_threshold):
                linked_mat_row.append(0)
            else:
                linked_mat_row.append(-1)
        linked_mat.append(linked_mat_row)

    # p[i][j]==-1  ---> 0/255
    for i in range(1, n-1):
        for j in range(1, m-1):
            if(linked_mat[i][j] == -1):
                if(linked_mat[i][j+1] == 255 or linked_mat[i][j-1] == 255 or linked_mat[i+1][j] == 255 or linked_mat[i-1][j] == 255
                   or linked_mat[i+1][j+1] == 255 or linked_mat[i+1][j-1] == 255 or linked_mat[i-1][j-1] == 255 or linked_mat[i-1][j+1] == 255):
                    linked_mat[i][j] = 255
                else:
                    linked_mat[i][j] = 0
    return linked_mat

# kernel_NXN : size of kernel
# high,low : high,low threshold ratios


def edge_detect(filtered_img, high, low, kernel_NXN, sigma):
    kernel_NXN = kernel_NXN
    sigma = sigma
    # filtered_img=gaussian_filter(input_img,sigma,kernel_NXN)
    strength_mat, direction_mat = gradient(filtered_img)
    suppressed_img = non_maxima_suppression(strength_mat, direction_mat)
    linked_img = dual_threshold(suppressed_img, high, low)
    return np.array(linked_img)
