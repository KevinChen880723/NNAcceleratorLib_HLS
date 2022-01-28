# HLS_MNIST


## Introduction to the Project

This is the hello world project to HLS; I implemented the MNIST inference device and deployed it on PYNQ-z2. In the project, I hand-crafted some core NN APIs: **Convolution**, **ReLU**, **Max-Pooling**, and **Fully connected** layers by HLS (High-Level Synthesis).

## NN Architecture

![](https://i.imgur.com/uzVrweO.png)

## Architecture

For the interface between PS and PL, I used AXI-Master interface. The system was developed by the concept of <span style="color:orange">dataflow</span>, which enables the kernel to run efficiently (Each layer does not need to wait until the previous module finished). In order to transfer the array data received from AXI-Master to the dataflow stream, I used ReadMem and WriteToMem blocks to transform.

![](https://i.imgur.com/NYfLbGR.png)

### Convolutional Layers

In order to make the convolutional layer run in a streaming way, I used the concept of "Line Buffer" to implement the one-channel convolution. In this way, the convolutional layer can process the data whenever it receives a new pixel. After building the one-channel convolution, I implement three-dimensional convolution by sequentially working on every channel.

![](https://i.imgur.com/nOHPVmK.png)

## Block Design

![](https://i.imgur.com/tSPEWmu.png)

## Inference Results

PYNQ-z2 spent <span style="color:red">1.5ms</span> to inference on an image and successfully got the correct prediction. Except for the one image testing, I also tested the project on the testing set, and the accuracy achieved <span style="color:red">98.55667%</span>!

![](https://i.imgur.com/eedfY8C.png)
