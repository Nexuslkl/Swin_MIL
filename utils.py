import torch
import cv2
import numpy as np
from skimage import measure


def gm(tensor, r):
    num = tensor.shape[0]
    output = torch.zeros(num, 1)
    for i in range(num):
        num_tensor = tensor[i].numel()
        tensor_r = tensor[i]**r
        sum_tensor = tensor_r.sum()
        average_tensor = sum_tensor/num_tensor
        pos_rate = (average_tensor+0.000001)**(1/r)
        output[i] = pos_rate
    return output


def hausdorff(a, b):
    hausdorff_sd = cv2.createHausdorffDistanceExtractor()
    contours_a, hierarchy_a = cv2.findContours(a.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours_b, hierarchy_b = cv2.findContours(b.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    temp1 = None
    for i in range(len(contours_a)):
        if i == 0:
            temp1 = contours_a[0]
        else:
            temp1 = np.concatenate((temp1, contours_a[i]), axis=0)
    if temp1 is not None:
        contours_a = temp1
    else:
        contours_a = np.zeros((1, 1, 2), dtype=int)

    temp2 = None
    for i in range(len(contours_b)):
        if i == 0:
            temp2 = contours_b[0]
        else:
            temp2 = np.concatenate((temp2, contours_b[i]), axis=0)
    if temp2 is not None:
        contours_b = temp2
    else:
        contours_b = np.zeros((1, 1, 2), dtype=int)

    hausdorff_distance = hausdorff_sd.computeDistance(contours_a, contours_b)
    return hausdorff_distance


def minmaxscaler(data):
    mindata = data.min()
    maxdata = data.max()
    return (data - mindata)/(maxdata-mindata)

