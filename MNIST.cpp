#include <hls_stream.h>
#include "./MNIST.h"
#include "conv1.h"

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

void readMemory(
		datatype 	  	      *img,
		unsigned short	      width_input,
		unsigned short	      height_input,
		hls::stream<datatype> &pixel_stream)
{
	for (int i = 0; i < width_input*height_input; i++){
		pixel_stream.write(img[i]);
	}
}

void WriteToMem(
		unsigned short 			  num_channel,
        unsigned short            width,
        unsigned short            height,
        hls::stream<datatype>     &pixel_stream,
        datatype                  *dst)
{
    write_image: for (int n = 0; n < num_channel*height*width; n++) {
        datatype pix = pixel_stream.read();
        dst[n] = pix;
    }
}

void test(datatype *img, datatype *output){
#pragma HLS interface s_axilite port=return
#pragma HLS interface m_axi depth=50 port=img
#pragma HLS interface m_axi depth=50 port=output
#pragma HLS dataflow
	myStream input_stream, output_stream;
	readMemory(img, IMAGE_WIDTH, IMAGE_HEIGHT, input_stream);
	conv1(Wconv1, Bconv1, 28, 28, input_stream, output_stream);
	WriteToMem(10, 26, 26, output_stream, output);
	return;
}
