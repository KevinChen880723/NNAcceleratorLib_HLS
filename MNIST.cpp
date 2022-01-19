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
	//Create Objects of the NN Layer
	YKHLS::Conv2D<FILTER_H_SIZE, FILTER_V_SIZE, layer1ChannelNum, layer2ChannelNum> conv1(28, 28, 26, 26);
	YKHLS::ReLU relu1(layer2ChannelNum, 26, 26);
	YKHLS::Conv2D<FILTER_H_SIZE, FILTER_V_SIZE, layer2ChannelNum, layer3ChannelNum> conv2(26, 26, 24, 24);
	YKHLS::ReLU relu2(layer3ChannelNum, 24, 24);

	// Declare hls::stream object which will used later
	myStream conv1_input_stream("conv1_input_stream");
	myStream conv1_output_stream("conv1_output_stream");
	myStream conv2_input_stream("conv2_input_stream");
	myStream conv2_output_stream("conv2_output_stream");
	myStream output_stream("output_stream");

	// Main Procedures of Inferencing
	readMemory(img, IMAGE_WIDTH, IMAGE_HEIGHT, conv1_input_stream);
	conv1(Wconv1, Bconv1, conv1_input_stream, conv1_output_stream);
	relu1(conv1_output_stream, conv2_input_stream);
	conv2(Wconv2, Bconv2, conv2_input_stream, conv2_output_stream);
	relu2(conv2_output_stream, output_stream);
	WriteToMem(layer3ChannelNum, 24, 24, output_stream, output);
	return;
}
