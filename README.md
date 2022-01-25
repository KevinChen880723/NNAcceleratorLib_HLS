# HLS_MNIST


## Introduction to the Project

This is the hello world project to HLS, I implemented MNIST inference device and deployed on PYNQ-z2. In the project, I hand-crafted some core NN API: **Convolution**, **ReLU**, **Max-Pooling**, and **Fully connected layers** by HLS (High Level Synthesis). 

## NN Architecture

![](https://i.imgur.com/uzVrweO.png =80%x)

## Architecture

The system developed by the concept of dataflow, which can enable the kernel run efficiently (Each layer no need to wait for previous module finished). For the interface between PS and PL, I used AXI-Master interface. In order to transfer the array data received from AXI-Master to dataflow stream, I used `ReadMem` and `WriteToMem` blocks to make the transformation.

![](https://i.imgur.com/NYfLbGR.png)

### Convolutional Layers

In order to make the convolutional layer run in the streaming way, I used the concept of "Line Buffer" to implement the one-channel convolution. In this way, the convolutional layer can process the data whenever it received a new pixel. After building the one-channel convolution, I implement three-dimensional convolution by sequentially work on every channels.

![](https://i.imgur.com/nOHPVmK.png)

## Block Design

![](https://i.imgur.com/tSPEWmu.png)

## Inference Results

PYNQ-z2 spent 1.5ms to inference on an image and successfully got the correct prediction. Except to the one image testing, I also tested the project on the testing set, the accuracy achieved 98.55667%!

![](https://i.imgur.com/eedfY8C.png)
