#include <hls_stream.h>
#include "MNIST.h"

#ifndef __SYNTHESIS__
//	#define PRINT
#endif

void readMemory(
		myDatatype 	  	      *img,
		unsigned short	      width_input,
		unsigned short	      height_input,
		hls::stream<myDatatype> &pixel_stream)
{
#pragma HLS interface ap_ctrl_none port=return
	for (int i = 0; i < width_input*height_input; i++){
		myDatatype data = img[i];
		pixel_stream.write(data);
		#ifdef PRINT
			std::cout << "First read: " << img[i] << "(" << i << ")" << std::endl;
		#endif
	}
}

void WriteToMem(
		unsigned short 			  num_channel,
        unsigned short            width,
        unsigned short            height,
        hls::stream<myDatatype>     &pixel_stream,
		myDatatype                  *dst)
{
#pragma HLS interface ap_ctrl_none port=return
    write_image: for (int n = 0; n < num_channel*height*width; n++) {
    	myDatatype pix = pixel_stream.read();
		#ifdef PRINT
        	std::cout << "pix in WriteToMem: " << pix << std::endl;
		#endif
        dst[n] = pix;
    }
}

void MNIST(myDatatype *img, myDatatype *output){
#pragma HLS interface ap_ctrl_none port=return
#pragma HLS interface m_axi depth=50 port=img
#pragma HLS interface m_axi depth=50 port=output
#pragma HLS dataflow

	const myDatatype Wconv1[layer2ChannelNum][layer1ChannelNum][FILTER_V_SIZE*FILTER_H_SIZE] = {
		#include "./weight/conv1.weight.h"
	};
	const myDatatype Bconv1[layer2ChannelNum] = {
		#include "./weight/conv1.bias.h"
	};
	const myDatatype Wconv2[layer3ChannelNum][layer2ChannelNum][FILTER_V_SIZE*FILTER_H_SIZE] = {
		#include "./weight/conv2.weight.h"
	};
	const myDatatype Bconv2[layer3ChannelNum] = {
		#include "./weight/conv2.bias.h"
	};
	const myDatatype Wconv3[layer4ChannelNum][layer3ChannelNum][FILTER_V_SIZE*FILTER_H_SIZE] = {
		#include "./weight/conv3.weight.h"
	};
	const myDatatype Bconv3[layer4ChannelNum] = {
		#include "./weight/conv3.bias.h"
	};
	const myDatatype Wconv4[layer5ChannelNum][layer4ChannelNum][FILTER_V_SIZE*FILTER_H_SIZE] = {
		#include "./weight/conv4.weight.h"
	};
	const myDatatype Bconv4[layer5ChannelNum] = {
		#include "./weight/conv4.bias.h"
	};
	const myDatatype Wfc4[FC1OutfeatNum][FC1InfeatNum] = {
		#include "./weight/fc.weight.h"
	};
	const myDatatype Bfc4[FC1OutfeatNum] = {
		#include "./weight/fc.bias.h"
	};

	//Create Objects of the NN Layer
	YKHLS::Conv2D<FILTER_H_SIZE, FILTER_V_SIZE, layer1ChannelNum, layer2ChannelNum, 28, 28, 26, 26> conv1;
	YKHLS::ReLU relu1(layer2ChannelNum, 26, 26);
	YKHLS::Conv2D<FILTER_H_SIZE, FILTER_V_SIZE, layer2ChannelNum, layer3ChannelNum, 26, 26, 24, 24> conv2;
	YKHLS::ReLU relu2(layer3ChannelNum, 24, 24);
	YKHLS::MaxPool2D<2, 2, 24, 24, layer3ChannelNum> maxpool1;
	YKHLS::Conv2D<FILTER_H_SIZE, FILTER_V_SIZE, layer3ChannelNum, layer4ChannelNum, 12, 12, 10, 10> conv3;
	YKHLS::ReLU relu3(layer4ChannelNum, 10, 10);
	YKHLS::Conv2D<FILTER_H_SIZE, FILTER_V_SIZE, layer4ChannelNum, layer5ChannelNum, 10, 10, 8, 8> conv4;
	YKHLS::ReLU relu4(layer5ChannelNum, 8, 8);
	YKHLS::MaxPool2D<2, 2, 8, 8, layer5ChannelNum> maxpool2;
	YKHLS::Linear2D<10, 64> fc;

	// Declare hls::stream object which will used later
	myStream conv1_input_stream("conv1_input_stream");
	myStream conv1_output_stream("conv1_output_stream");
	myStream conv2_input_stream("conv2_input_stream");
	myStream conv2_output_stream("conv2_output_stream");
	myStream maxpool1_input_stream("maxpool1_input_stream");
	myStream maxpool1_output_stream("maxpool1_output_stream");
	myStream conv3_output_stream("conv3_output_stream");
	myStream conv4_input_stream("conv4_input_stream");
	myStream conv4_output_stream("conv4_output_stream");
	myStream relu4_output_stream("relu4_output_stream");
	myStream fc_input_stream("fc_input_stream");
	myStream output_stream("output_stream");

	// Main Procedures of Inferencing
	readMemory(img, IMAGE_WIDTH, IMAGE_HEIGHT, conv1_input_stream);
	conv1(Wconv1, Bconv1, conv1_input_stream, conv1_output_stream);
	relu1(conv1_output_stream, conv2_input_stream);
	conv2(Wconv2, Bconv2, conv2_input_stream, conv2_output_stream);
	relu2(conv2_output_stream, maxpool1_input_stream);
	maxpool1(maxpool1_input_stream, maxpool1_output_stream);
	conv3(Wconv3, Bconv3, maxpool1_output_stream, conv3_output_stream);
	relu3(conv3_output_stream, conv4_input_stream);
	conv4(Wconv4, Bconv4, conv4_input_stream, conv4_output_stream);
	relu4(conv4_output_stream, relu4_output_stream);
	maxpool2(relu4_output_stream, fc_input_stream);
	fc(Wfc4, Bfc4, fc_input_stream, output_stream);
	WriteToMem(1, 1, 10, output_stream, output);
	return;
}
