#ifndef MNIST
#define MNIST

#define layer1CnannelNum 1
#define layer2CnannelNum 10
#define layer3CnannelNum 8
#define layer4CnannelNum 6
#define layer5CnannelNum 4
#define FC1InfeatNum 64
#define FC1OutfeatNum 10
typedef hls::stream<float> myStream;
void test(int a, int b, int *c);

#endif