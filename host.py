from __future__ import print_function

import sys
import numpy as np
from time import time 
import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2

sys.path.append('/home/xilinx')
from pynq import Overlay
from pynq import allocate

def ap_fixed_converter(ap_fixed, total_bits, frac_bits): 
    mask1 = 1 << (total_bits - 1) 
    mask2 = mask1 - 1 
    return ((ap_fixed & mask2) - (ap_fixed & mask1)) / (1 << frac_bits)  

if __name__ == "__main__":
    overlay_mnist = Overlay("./MNIST.bit")
    mnist = overlay_mnist.MNIST_0

    gt_list = []
    with open("./testData/gt.txt", "r") as file:
        for i in range(60000):
            gt_list.append(int(file.readline()))
    
    match = 0
    
    timeKernelStart = time()
    for i in range(60000):
        pixels = list(cv2.imread("./testData/img/{}.jpg".format(i), cv2.IMREAD_GRAYSCALE).reshape(784))
#     testImg = open("input.txt", "r+")
#     pixels = []
#     for i in range(784):
#         pixels.append(int(float(testImg.readline()[:-1])))
    
#     img = np.zeros((28, 28))
#     for y in range(28):
#         for x in range(28):
#             img[y][x] = pixels[y*28+x]
#     img = Image.fromarray(img)
#     plt.imshow(img)
#     plt.show()

        inBuffer0 = allocate(shape=(784,), dtype=np.uint8)
        outBuffer0 = allocate(shape=(10,), dtype=np.uint16)
        for j in range(784):
            inBuffer0[j] = pixels[j]

        mnist.write(0x10, inBuffer0.device_address)
        mnist.write(0x1C, outBuffer0.device_address)
        mnist.write(0x00, 0x01)
        while (mnist.read(0x00) & 0x4) == 0x0:
            continue
        output = np.zeros(10)
        for j in range(10):
            output[j] = ap_fixed_converter(outBuffer0[j], 16, 6)
        prediction = np.argmax(output)
        if prediction == gt_list[i]:
            match += 1
#         print("Prediction is: {}".format(prediction))
        if i % 100 == 0:
            print(i)
    
    timeKernelEnd = time()
    print("Elapsed time: {} s".format(timeKernelEnd-timeKernelStart))
    print("match = {}".format(match))
    print("Accuracy: {}%".format(match/60000))