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
    for i in range(n):
      input_img[i] = [0]*wrap_size + input_img[i] + [0]*wrap_size
    for h in range(wrap_size):
      input_img = [[0]*(m+2*wrap_size)] + input_img + [[0]*(m+2*wrap_size)]
    for j in range(wrap_size): 
      input_img[j] = input_img[(wrap_size+1)*-1]
      input_img[(j+1)*-1] = input_img[(wrap_size+1)*-1]
    for k in range(len(input_img)-1):
      for j in range(wrap_size):
        input_img[k][j] = input_img[k][(wrap_size+1)*-1]
        input_img[k][(j+1)*-1] = input_img[k][(wrap_size+1)*-1]
    return input_img