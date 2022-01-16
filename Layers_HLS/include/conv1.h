#ifndef CONV1
#define CONV1

#include "ap_int.h"
#include "hls_stream.h"
#include "common.h"
#include "MNIST.h"
void ReadFromMem(
        unsigned short            width,
        unsigned short            height,
        datatype                  *weights,
        hls::stream<datatype>     &input_stream,
        hls::stream<datatype>     &coeff_stream,
        hls::stream<datatype>     &pixel_stream );

void Window2D(
        unsigned short          width,
        unsigned short          height,
        hls::stream<datatype>   &pixel_stream,
        hls::stream<window>     &window_stream,
        ap_int<1>               do_padding);

void Filter2D(
        unsigned short              width,
        unsigned short              height,
        hls::stream<datatype>       &coeff_stream,
        hls::stream<window>         &window_stream,
		hls::stream<datatype>       &output_stream,
        ap_int<1>                   do_padding);        

void Filter2DKernel(
        datatype                 *Wconv,
        unsigned short           width_input,
        unsigned short           height_input,
        hls::stream<datatype>    &input_stream,
        hls::stream<datatype>    &output_stream);

void summation(
        datatype                 bias,
        unsigned short           width,
        unsigned short           height,
        unsigned short           num_channel,
        hls::stream<datatype>    &ChannelOutput_stream,
        hls::stream<datatype>    &OverallOutput_stream);

void conv1(
        datatype                 Wconv[layer2CnannelNum][layer1CnannelNum][FILTER_V_SIZE*FILTER_H_SIZE],
        datatype                 Bconv[layer2CnannelNum],
        unsigned short           width_input,
        unsigned short           height_input,
        hls::stream<datatype>    &input_stream,
        hls::stream<datatype>    &OverallOutput_stream);

#endif