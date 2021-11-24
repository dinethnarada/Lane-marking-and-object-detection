import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import gray_scale
from canny_edge_detector import edge_detect,gaussian_filter
from transform import resize
from normalize import normalize_intensity
from filters import filter_func
from hough_transform import hough_transform,superimpose

# Function - Display the images
# Return - None
def image_display(image, window_name):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function - Load the images from /images folder
# Return - List containing Image Paths
def read_imgs(path):
    print('Default kernel size - 3')
    print('Loading All the images from', path, 'folder')
    return [img_path for img_path in os.listdir(path)
            if img_path.split(".")[-1].lower() == "jpeg"
            or img_path.split(".")[-1].lower() == "jpg"
            ]

# Function - Load img and convert it to gray image scale
# Return - Grey image
def load_img(img_path,path):
    original_img = cv2.imread(path+'/'+img_path)
    func_img = gray_scale(original_img)
    # Display gray-scale images
    gray_img = np.array(func_img, dtype=np.uint8)
    print("original shape",gray_img.shape)
    image_display(gray_img, "Original image")

    input_img = func_img.tolist()
    return gray_img,input_img

def save_imgs(image, filename, img_format,file_type):
    if(file_type!='normalized' and file_type!='median' and file_type!='guassian'):
        output_img = np.array(image, dtype=np.uint8)
    else:
        output_img = np.array(image)
    window_name = "Detected Image"
    image_display(output_img, window_name)

    current_dir = os.path.dirname(os.path.realpath(__file__))
    filename = current_dir+"/output/"+file_type+"."+img_format
    #cv2.imwrite(filename, output_img)
    if(file_type!='hough'):
        plt.imsave(filename,image,cmap='gray')
    else:
        cv2.imwrite(filename, output_img)

# Function - Main Function
def main_170050R():
    path = "images"
    img_paths = read_imgs(path)
    print(img_paths)
    for img_path in img_paths:
        img_name, img_format = img_path.split(".")
        # Step 1 - Grey Scale
        gray_img, input_img = load_img(img_path,path)

        # Step 2 - Scale the image by 80%
        scale_img = resize(input_img, 0.8)
        print("shape",np.array(scale_img).shape)
        save_imgs(scale_img, img_name, img_format,"scaled")

        # Step 3 - Normalize the intensity
        normalized_img = normalize_intensity(scale_img)
        print("shape",np.array(normalized_img).shape)
        save_imgs(normalized_img, img_name, img_format,"normalized")
        
        # Step 4 - Apply Median Filter
        median_filter_img = filter_func(normalized_img,'median')
        print("median shape",np.array(median_filter_img).shape)
        save_imgs(median_filter_img, img_name, img_format,"median")

        # Step 5 - Apply Guassian Mean Filter
        filtered_img = gaussian_filter(median_filter_img,sigma=1,kernel_NXN=3)
        print("guassian shape",np.array(filtered_img).shape)
        save_imgs(filtered_img, img_name, img_format,"guassian")

        # Step 6 - Canny Edge Detector
        edge_detected_img=edge_detect(filtered_img,0.15,0.05,3,1)
        print("canny shape",np.array(edge_detected_img).shape)
        save_imgs(edge_detected_img, img_name, img_format,"edge")

        # Step 7 - Apply Hough Transform
        indicies,rhos,thetas = hough_transform(edge_detected_img)

        # Step 8 - Superimpose the lanes by red color
        superimpose_img = superimpose(indicies,rhos,thetas)
        print("hough shape",np.array(superimpose_img).shape)
        save_imgs(superimpose_img, img_name, img_format,"Lane_170050R")
    return


def main():
    main_170050R()


if __name__ == "__main__":
    main()
