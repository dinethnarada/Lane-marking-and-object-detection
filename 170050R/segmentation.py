import numpy as np
import cv2
import os

# Function - get four or eight neighbors
# Params - Neigbror number(user)
# Return - return connets
def selectNeighbours(neighbor_num):
    if neighbor_num == 8:
        connects = [[-1,-1],[0,-1],[1,-1],[1,0],[1,1],[0,1],[-1,1],[-1,0]]
    elif neighbor_num == 4:
        connects = [[0,1],[1,0],[0,1],[-1,0]]
    else:
        raise ValueError("The neighbor_num should be 4 or 8")
    return connects

# Function - Single seed region grow algorithm
# Params - Image, mask, Seed, Thresold, neighbour number, labels
# Return - Identified Regions upto now
def regionGrow(img, mask, seed, thresh, neighbor_num=8, label=1):
    height, weight = img.shape

    connects = selectNeighbours(neighbor_num)

    seedList = []
    seedList.append(seed)

    while (len(seedList) > 0):
        cPoint = seedList.pop(0)
        mask[cPoint[0], cPoint[1]] = label
        for i in range(neighbor_num):
            tmpX = cPoint[0] + connects[i][0]
            tmpY = cPoint[1] + connects[i][1]
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight or mask[tmpX, tmpY] != 0:
                continue
            grayDiff = abs(int(img[cPoint[0], cPoint[1]]) - int(img[tmpX, tmpY]))
            if  mask[tmpX, tmpY] == 0 and grayDiff < thresh:
                mask[tmpX, tmpY] = label
                seedList.append([tmpX, tmpY])
    return mask

# Function - Find out the non zero value indexes
# Params - mask
# Return - Next Seed coordinates
def next_seed(mask):
    h = len(mask)
    w = len(mask[0])
    l1 = []
    for eh in range(h):
        for ew in range(w):
            if mask[eh][ew] == 0:
                l1.append([eh,ew])
    if len(l1[0]) == 0:
        return None
    x = l1[0][0]
    y = l1[1][0]
    return [x,y]

# Function - base algorithm
def img_region_grow(img, label_in):
    """ gray image region grow algorithm, different region will have different labels """
    mask = np.array([[0 for x in range(img.shape[1])] for y in range(img.shape[0])])
    thresh = 10
    label = label_in

    while True:
        seed = next_seed(mask)
        if seed is not None:
            mask = regionGrow(img, mask, seed, thresh, neighbor_num=8, label=label)
            label += 10
        else:
            print("Process Done!")
            break
    return mask


def segmentation():
    img = cv2.imread('output/guassian.jpg', 0)  # read in gray image mode
    mask = img_region_grow(img, 1)
    mask  = np.array(mask, dtype=np.uint8)
    
    # Save Segment Image
    current_dir = os.path.dirname(os.path.realpath(__file__))
    filename = current_dir+"/output/"+"Segment_170050R.jpg"
    cv2.imwrite(filename, mask)

    # Show Segmented Image
    cv2.imshow("final",mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    segmentation()