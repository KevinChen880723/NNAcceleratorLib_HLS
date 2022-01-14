#include <hls_stream.h>
#include "./MNIST.h"

const float Wconv1[layer2CnannelNum][layer1CnannelNum][3][3] = {
	#include "./weight/conv1.weight.h"
};
const float Bconv1[layer2CnannelNum] = {
	#include "./weight/conv1.bias.h"
};
const float Wconv2[layer3CnannelNum][layer2CnannelNum][3][3] = {
	#include "./weight/conv2.weight.h"
};
const float Bconv2[layer3CnannelNum] = {
	#include "./weight/conv2.bias.h"
};
const float Wconv3[layer4CnannelNum][layer3CnannelNum][3][3] = {
	#include "./weight/conv3.weight.h"
};
const float Bconv3[layer4CnannelNum] = {
	#include "./weight/conv3.bias.h"
};
const float Wconv4[layer5CnannelNum][layer4CnannelNum][3][3] = {
	#include "./weight/conv4.weight.h"
};
const float Bconv4[layer5CnannelNum] = {
	#include "./weight/conv4.bias.h"
};
const float Wfc4[FC1OutfeatNum][FC1InfeatNum] = {
	#include "./weight/fc.weight.h"
};
const float Bfc4[FC1OutfeatNum] = {
	#include "./weight/fc.bias.h"
};


void test(myStream &img, myStream &output){
#pragma HLS interface s_axilite port=return
#pragma HLS interface axis register both port=img
#pragma HLS interface axis register both port=output
#pragma HLS dataflow
	*c = a + b;
	return;
}
