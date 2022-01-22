#include "../include/ReLU.h"
#define RUN_CO_SIM
namespace YKHLS{
	ReLU::ReLU(): channel_num(1), width(1), height(1){

	}

	ReLU::ReLU(unsigned short c, unsigned short w, unsigned short h): channel_num(c), width(w), height(h){

	}

	void ReLU::operator()(
				hls::stream<myDatatype>    	&input_stream,
				hls::stream<myDatatype>    	&output_stream)
	{
#ifndef RUN_CO_SIM
#pragma HLS interface ap_ctrl_none port=return
#endif
		for (int i = 0; i < channel_num * height * width; i++){
			myDatatype temp = input_stream.read();
			output_stream.write((temp < 0)? myDatatype(0): temp);
		}
	}
}
