# To-do list
- Solve the bottleneck in the convolutional layer: The bottleneck causes the significant usage of BRAM because the deep FIFO is needed..
- Use Pipeline strategy in the fully connected layer: The current buffer in the FC module can not be completely partitioned, and this causes the limitation of the pipeline.
- Introduce the Ping-Pong buffer and burst transfer in `ReadMem`: Access global memory introduces significant overhead, use Ping-Pong buffer to make the module read and write at the same time.
- Quantize the PyTorch model: The current project uses float32 weight and casts the type into `ap_fixed<16, 10, AP_RND_ZERO, AP_SAT>`. Although current latency is fast enough, if we can store the data as the lower precision data type, the device can run even faster and maintain high accuracy.

# Current Performance

![](https://i.imgur.com/jTsfJki.png)
