import math
import cv2

# Function - Calculate Non zero values indexes list
# Params - Image
def non_zero_index_list(image):
    h = len(image)
    w = len(image[0])
    l1 = []
    for eh in range(h):
        for ew in range(w):
            if image[eh][ew] != 0: # Find out the non zero value indexes
                l1.append([eh,ew])
    return l1

# Function - Calculate maximum values in list
# Params - List
# Return - indexes of maximum value in list
def find_maximun(H):
    current = 0
    for x in range(len(H)):
        for y in range(len(H[0])):
            if H[x][y] > current:
                current = H[x][y]
                ix,iy = x,y
    return ix,iy

# Function - Create the hough space
# Params - Canny edges list, non zero values list
# Return - Hough Space, rhos list and theta list
def hough_line_acc(canny, non_zero_index_list):
    height, width = len(canny), len(canny[0])
    diagonal = math.ceil(math.sqrt(height**2+width**2))
    rhos = [x for x in range(-diagonal, diagonal)]
    thetas = [math.radians(x) for x in range(-90,90)]

    # Cache the sin,cos values
    sin_array = [math.sin(x) for x in thetas]
    cos_array = [math.cos(x) for x in thetas]
    len_rhos = len(rhos)
    len_thetas = len(thetas)

    H = [[0 for x in range(len_thetas)] for y in range(len_rhos)]

    for i in range(len(non_zero_index_list)):
        y_val = non_zero_index_list[i][0]
        x_val = non_zero_index_list[i][1]
        for j in range(len_thetas):
            c_rho = int((x_val*cos_array[j])+(y_val*sin_array[j]) + diagonal)
            H[c_rho][j] += 1
    return H,thetas,rhos

# Function - Calculate Peak values in the hough space
# Params - Hough Space, Expected peaks, thresold, neigbourhood size
# Return - Hough space and Peak indices List
def hough_peaks(H, num_peaks, neigbour_size=3):
    indicies = []
    newH = H.copy()
    for i in range(num_peaks):
        # find index pais that gives maximum value in array
        maxX,maxY = find_maximun(newH)
        indicies.append([maxX,maxY])

        # split x,y coordinates
        idx_y, idx_x = maxX, maxY 
        if (idx_x - (neigbour_size/2)) < 0: min_x = 0
        else: min_x = idx_x - (neigbour_size/2)

        if ((idx_x + (neigbour_size/2) + 1) > len(H[0])): max_x = len(H[0])
        else: max_x = idx_x + (neigbour_size/2) + 1

        if (idx_y - (neigbour_size/2)) < 0: min_y = 0
        else: min_y = idx_y - (neigbour_size/2)

        if ((idx_y + (neigbour_size/2) + 1) > len(H)):max_y = len(H)
        else: max_y = idx_y + (neigbour_size/2) + 1

        for x in range(int(min_x), int(max_x)):
            for y in range(int(min_y), int(max_y)):
                newH[y][x] = 0
                if (x == min_x or x == (max_x - 1)):
                    H[y][x] = 255
                if (y == min_y or y == (max_y - 1)):
                    H[y][x] = 255
    print(indicies)
    return indicies, H

# Function - Recongnize the Linear function parameters and Draw the red lines
# Params - Image, Peak indicies list, rhos list, thetas list
# Return - None
def hough_lines(img, indicies, rhos, thetas):
    ''' A function that takes indicies a rhos table and thetas table and draws
        lines on the input images that correspond to these values. '''
    lines = []
    for i in range(len(indicies)):
        rho = rhos[indicies[i][0]]
        theta = thetas[indicies[i][1]]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        lines.append([x1, y1, x2, y2])
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Function - Apply hough transform to image
# Params - Canny edges list
# Return - Indices, rhos list and thetas list
def hough_transform(canny_img):
    n_list = non_zero_index_list(canny_img)
    H, thetas, rhos = hough_line_acc(canny_img,n_list)
    indicies, H = hough_peaks(H, 3, neigbour_size=11) 
    return indicies,rhos,thetas
    
# Function - Superimpose the road lines with scaled images
# Params - Peak indices, rhos list, thetas list
# Return - Superimposed image with red lines
def superimpose(indicies, rhos, thetas):
    shapes = cv2.imread('output/scaled.jpg')
    hough_lines(shapes, indicies, rhos, thetas)
    return shapes