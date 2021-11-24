import os
import cv2
import numpy as np

def select_sub_mat(wrapped_img,i,j,kernel_NXN=3):
  sub_mat=[]
  for r in range(i,i+kernel_NXN):
      sub_mat.append(wrapped_img[r][j:j+kernel_NXN])
  return sub_mat

def gray_scale(img):
    r,g,b=img[:,:,0],img[:,:,1],img[:,:,2]
    return 0.2989*r+0.5870*g+0.1140*b

def wrap(input_img, n, m, wrap_size):
    wrapped_img = []
    for r in range(n):
        wrapped_row = [input_img[r][-1]]*wrap_size + input_img[r]+[input_img[r][-1]]*wrap_size
        wrapped_img.append(wrapped_row)
    for iter in range(wrap_size):
        wrapped_img.insert(0, wrapped_img[-1])
        wrapped_img.append(wrapped_img[0])
    return wrapped_img