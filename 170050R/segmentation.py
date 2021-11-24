import numpy as np
import cv2
import os


# get the difference of gray value in image
def getGrayDiff(img, currentPoint, tmpPoint):
    return abs(int(img[currentPoint[0], currentPoint[1]]) - int(img[tmpPoint[0], tmpPoint[1]]))


# get four or eight neighbors
def selectConnects(neighbor_num):
    if neighbor_num == 8:
        connects = [[-1,-1],[0,-1],[1,-1],[1,0],[1,1],[0,1],[-1,1],[-1,0]]
    elif neighbor_num == 4:
        connects = [[0,1],[1,0],[0,1],[-1,0]]
    else:
        raise ValueError("The neighbor_num should be 4 or 8")
    return connects


def regionGrow(img, mask, seed, thresh, neighbor_num=8, label=1):
    """ single seed region grow algorithm """
    height, weight = img.shape

    connects = selectConnects(neighbor_num)

    seedList = []
    seedList.append(seed)

    while (len(seedList) > 0):
        currentPoint = seedList.pop(0)

        mask[currentPoint[0], currentPoint[1]] = label
        for i in range(neighbor_num):
            tmpX = currentPoint[0] + connects[i][0]
            tmpY = currentPoint[1] + connects[i][1]
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight or mask[tmpX, tmpY] != 0:
                continue
            grayDiff = getGrayDiff(img, currentPoint, [tmpX, tmpY])
            if grayDiff < thresh and mask[tmpX, tmpY] == 0:
                mask[tmpX, tmpY] = label
                seedList.append([tmpX, tmpY])
    return mask


def find_undetermined(mask):
    h = len(mask)
    w = len(mask[0])
    l1 = []
    for eh in range(h):
        for ew in range(w):
            if mask[eh][ew] == 0: # Find out the non zero value indexes
                l1.append([eh,ew])
    if len(l1[0]) == 0:
        return None
    x = l1[0][0]
    y = l1[1][0]

    return [x,y]


def img_region_grow(img, label_in):
    """ gray image region grow algorithm, different region will have different labels """
    mask = np.array([[0 for x in range(img.shape[1])] for y in range(img.shape[0])])
    #mask = np.zeros(img.shape)
    thresh = 10
    label = label_in

    while True:
        seed = find_undetermined(mask)
        if seed is not None:
            mask = regionGrow(img, mask, seed, thresh, neighbor_num=8, label=label)
            # cv2.imshow("mask", mask)
            label += 10
        else:
            print("Process Done!")
            break
    return mask


def segmentation():
    img = cv2.imread('output/scaled-output-dashcam_view_1.jpg', 0)  # read in gray image mode
    mask = img_region_grow(img, 1)
    mask  = np.array(mask, dtype=np.uint8)
 
    current_dir = os.path.dirname(os.path.realpath(__file__))
    filename = current_dir+"/output/"+"Segment_170050R.jpg"
    cv2.imwrite(filename, mask)

    cv2.imshow("final",mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    segmentation()