#ifndef CONV1
#define CONV1

#include "ap_int.h"
#include "hls_stream.h"
#include "../../common.h"

struct window {
    myDatatype pix[FILTER_V_SIZE][FILTER_H_SIZE];
};


void ReadFromMem(
        unsigned short            width,
        unsigned short            height,
		const myDatatype                  weights[FILTER_V_SIZE*FILTER_H_SIZE],
        hls::stream<myDatatype>     &input_stream,
        hls::stream<myDatatype>     &coeff_stream,
        hls::stream<myDatatype>     &pixel_stream );

void Window2D(
        unsigned short          width,
        unsigned short          height,
        hls::stream<myDatatype>   &pixel_stream,
        hls::stream<window>     &window_stream,
        ap_int<1>               do_padding);

void Filter2D(
        unsigned short              width,
        unsigned short              height,
        hls::stream<myDatatype>       &coeff_stream,
        hls::stream<window>         &window_stream,
		hls::stream<myDatatype>       &output_stream,
        ap_int<1>                   do_padding);        

void Filter2DKernel(
        myDatatype                 *Wconv,
        unsigned short           width_input,
        unsigned short           height_input,
        hls::stream<myDatatype>    &input_stream,
        hls::stream<myDatatype>    &output_stream);

void summation(
        myDatatype                 bias,
        unsigned short           width,
        unsigned short           height,
        unsigned short           num_channel,
        hls::stream<myDatatype>    &ChannelOutput_stream,
        hls::stream<myDatatype>    &OverallOutput_stream);

void pixelBuffer(
        unsigned short           width_input,
        unsigned short           height_input,
	    unsigned short 			 channel_input,
	    unsigned short 			 channel_output,
        hls::stream<myDatatype>    &input_stream,
        hls::stream<myDatatype>    &Buffer_stream);

void conv1(
		const myDatatype                 Wconv[layer2ChannelNum][layer1ChannelNum][FILTER_V_SIZE*FILTER_H_SIZE],
		const myDatatype                 Bconv[layer2ChannelNum],
        unsigned short           width_input,
        unsigned short           height_input,
        hls::stream<myDatatype>    &input_stream,
        hls::stream<myDatatype>    &OverallOutput_stream);

#endif

