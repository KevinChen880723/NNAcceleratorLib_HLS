#include "./common.h"
#include "./Layers_HLS/src/Conv2D.cpp"
#include "./Layers_HLS/include/Conv2D.h"
#include "./Layers_HLS/include/ReLU.h"
#include "./Layers_HLS/src/MaxPool2D.cpp"
#include "./Layers_HLS/src/Linear2D.cpp"
#include "./Layers_HLS/include/Linear2D.h"
//
//const myDatatype Wconv1[layer2ChannelNum][layer1ChannelNum][FILTER_V_SIZE*FILTER_H_SIZE] = {
//	#include "./weight/conv1.weight.h"
//};
//const myDatatype Bconv1[layer2ChannelNum] = {
//	#include "./weight/conv1.bias.h"
//};
//const myDatatype Wconv2[layer3ChannelNum][layer2ChannelNum][FILTER_V_SIZE*FILTER_H_SIZE] = {
//	#include "./weight/conv2.weight.h"
//};
//const myDatatype Bconv2[layer3ChannelNum] = {
//	#include "./weight/conv2.bias.h"
//};
//const myDatatype Wconv3[layer4ChannelNum][layer3ChannelNum][FILTER_V_SIZE*FILTER_H_SIZE] = {
//	#include "./weight/conv3.weight.h"
//};
//const myDatatype Bconv3[layer4ChannelNum] = {
//	#include "./weight/conv3.bias.h"
//};
//const myDatatype Wconv4[layer5ChannelNum][layer4ChannelNum][FILTER_V_SIZE*FILTER_H_SIZE] = {
//	#include "./weight/conv4.weight.h"
//};
//const myDatatype Bconv4[layer5ChannelNum] = {
//	#include "./weight/conv4.bias.h"
//};
//const myDatatype Wfc4[FC1OutfeatNum][FC1InfeatNum] = {
//	#include "./weight/fc.weight.h"
//};
//const myDatatype Bfc4[FC1OutfeatNum] = {
//	#include "./weight/fc.bias.h"
//};

void MNIST(ap_uint<8> *img, myDatatype *output);

