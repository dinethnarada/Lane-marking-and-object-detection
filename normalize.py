import math

def normalize_intensity(image):
    h,w = len(image), len(image[0])
    output_img = [[None]*w for _ in range(h)]
    sub_arr=[px for sub_row in image for px in sub_row] #load to array
    mean=sum(sub_arr)/(h*w)
    min_v = min(sub_arr)
    max_v = max(sub_arr)
    print(mean,min_v,max_v,(h*w))
    for i in range(len(output_img)):
        for j in range(len(output_img[0])):
            output_img[i][j] = (image[i][j] - mean)/(max_v-min_v)

    return output_img