#include "ap_int.h"
#include "hls_stream.h"
#include "../../common.h"

#ifndef CONV2DDEF
#define CONV2DDEF

namespace YKHLS{
	template<	unsigned short 				width_filter,
				unsigned short 				height_filter,
				unsigned short 				channel_input,
				unsigned short 				channel_output,
				unsigned short 				width_input,
				unsigned short 				height_input,
				unsigned short 				width_output,
				unsigned short 				height_output>
	class Conv2D{
	public:

		Conv2D();

		void operator()(
				const myDatatype            Wconv[channel_output][channel_input][height_filter*width_filter],
				const myDatatype            Bconv[channel_output],
				hls::stream<myDatatype>    	&input_stream,
				hls::stream<myDatatype>    	&OverallOutput_stream);
	private:
		int i;

		typedef struct window {
		    myDatatype pix[height_filter][width_filter];
		}window;

		void ReadFromMem(
				unsigned short            	width,
				unsigned short            	height,
				const myDatatype            weights[height_filter*width_filter],
				hls::stream<myDatatype>     &input_stream,
				hls::stream<myDatatype>     &coeff_stream,
				hls::stream<myDatatype>     &pixel_stream );

		void Window2D(
				unsigned short          	width,
				unsigned short          	height,
				hls::stream<myDatatype>   	&pixel_stream,
				hls::stream<window>     	&window_stream,
				ap_int<1>               	do_padding);

		void Filter2D(
				unsigned short              width,
				unsigned short              height,
				hls::stream<myDatatype>     &coeff_stream,
				hls::stream<window>         &window_stream,
				hls::stream<myDatatype>     &output_stream,
				ap_int<1>                   do_padding);

		void Filter2DKernel(
				const myDatatype            Wconv[height_filter*width_filter],
				hls::stream<myDatatype>     &input_stream,
				hls::stream<myDatatype>     &output_stream);

		void summation(
				myDatatype 					outputFeature[height_output][width_output],
				const myDatatype            bias,
				hls::stream<myDatatype>     &ChannelOutput_stream,
				hls::stream<myDatatype>     &OverallOutput_stream);

		void pixelBuffer(
				hls::stream<myDatatype>     &input_stream,
				hls::stream<myDatatype>     &Buffer_stream);

		void ConvolutionWithoutSum(
				const myDatatype                 Kernel[channel_input][height_filter*width_filter],
				hls::stream<myDatatype>    		 &Buffer_stream,
				hls::stream<myDatatype>    		 &ChannelOutput_stream);

		void ExecuteConv(
				const myDatatype                 Wconv[channel_output][channel_input][height_filter*width_filter],
				const myDatatype                 Bconv[channel_output],
				hls::stream<myDatatype>    		 &Buffer_stream,
				hls::stream<myDatatype>    		 &OverallOutput_stream);
	};
}

#endif
