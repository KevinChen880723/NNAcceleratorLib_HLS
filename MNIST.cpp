#include <hls_stream.h>
#include "MNIST.h"

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

void readMemory(
		myDatatype 	  	      *img,
		unsigned short	      width_input,
		unsigned short	      height_input,
		hls::stream<myDatatype> &pixel_stream)
{
//#pragma HLS interface ap_ctrl_none port=return
	std::cout << "The address of input_stream in readMemory() is: " << &pixel_stream << std::endl;
	for (int i = 0; i < width_input*height_input; i++){
		myDatatype data = img[i];
		pixel_stream.write(data);
//		myDatatype data2 = pixel_stream.read();
		std::cout << "First read: " << img[i] << "(" << i << ")" << std::endl;
//		data2 = pixel_stream.read();
//		std::cout << "Second read: " <<  data2 << std::endl;
//		data2 = pixel_stream.read();
//		std::cout << "Third Second read: " <<  data2 << std::endl;
	}
}

void WriteToMem(
		unsigned short 			  num_channel,
        unsigned short            width,
        unsigned short            height,
        hls::stream<myDatatype>     &pixel_stream,
		myDatatype                  *dst)
{
//#pragma HLS interface ap_ctrl_none port=return
    write_image: for (int n = 0; n < num_channel*height*width; n++) {
    	myDatatype pix = pixel_stream.read();
        std::cout << "pix in WriteToMem: " << pix << std::endl;
        dst[n] = pix;
//    	myDatatype pix;
//    	if (pixel_stream.read_nb(pix)){
//    		std::cout << "pix in WriteToMem: " << pix << std::endl;
//    		dst[n] = pix;
//    	}
//    	else{
//    		;
////    		std::cout << "failed"  << std::endl;
//    	}
    }
}

void MNIST(myDatatype *img, myDatatype *output){
#pragma HLS interface s_axilite port=return
#pragma HLS interface m_axi depth=50 port=img
#pragma HLS interface m_axi depth=50 port=output
#pragma HLS dataflow
	myStream input_stream("input_stream"), output_stream("output_stream");
	std::cout << "The address of input_stream in MNIST() is: " << &input_stream << std::endl;
	readMemory(img, IMAGE_WIDTH, IMAGE_HEIGHT, input_stream);
	conv1(Wconv1, Bconv1, 28, 28, input_stream, output_stream);
	WriteToMem(10, 26, 26, output_stream, output);
	return;
}
