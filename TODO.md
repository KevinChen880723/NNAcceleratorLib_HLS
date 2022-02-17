# To-do list
- [ ] Solve the bottleneck in the convolutional layer: The bottleneck causes the significant usage of BRAM because the deep FIFO is needed..
- [ ] Use Pipeline strategy in the fully connected layer: The current buffer in the FC module can not be completely partitioned, and this causes the limitation of the pipeline.
- [ ] Introduce the Ping-Pong buffer and burst transfer in `ReadMem`: Access global memory introduces significant overhead, use Ping-Pong buffer to make the module read and write at the same time.
- [ ] Quantize the PyTorch model: The current project uses float32 weight and casts the type into `ap_fixed<16, 10, AP_RND_ZERO, AP_SAT>`. Although current latency is fast enough, if we can store the data as the lower precision data type, the device can run even faster and maintain high accuracy.

# Current Performance

![](https://i.imgur.com/jTsfJki.png)

# Performance After Changing

## Using Pipeline Strategy In the Fully Connected Layer

### Resources Usage

The occupied resources was reduced after using pipeline strategy in the fully connected layer.

![](https://i.imgur.com/uJ23UQL.png)

### Inference Time

Since using the same kernel may have different inference times in various iterations. To compare the inference time fairly, I compare the kernel by the average elapsed time from 100,000 times. From the following results, we know that the latency of the two different kernels is almost the same.
- Pipelined fully connected layer: **1.46129232645034** mS
- Original kernel: **1.4616054916381837** mS

![](https://i.imgur.com/JsCTlow.png)

### Try to use the ping-pong buffer

![image](https://user-images.githubusercontent.com/55487740/154413855-f2063ee1-c1de-4fec-b8a2-7746fd7d75a2.png)
