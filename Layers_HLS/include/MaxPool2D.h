#include "ap_int.h"
#include "hls_stream.h"
#include "../../common.h"

namespace YKHLS{

	template< unsigned short 	width_filter,
			  unsigned short 	height_filter>
	class MaxPool2D{
	public:

		MaxPool2D();
		MaxPool2D(	unsigned short w_i,
					unsigned short h_i,
					unsigned short c_i);

		void operator()(
				hls::stream<myDatatype>    	&input_stream,
				hls::stream<myDatatype>     &output_stream);
	private:
		unsigned short            			width_input;
		unsigned short            			height_input;
		unsigned short            			channel_input;

		typedef struct window {
		    myDatatype pix[height_filter][width_filter];
		}window;

		void ReadFromMem(
				hls::stream<myDatatype>     &input_stream,
				hls::stream<myDatatype>     &pixel_stream);

		void Window2D(
				hls::stream<myDatatype>   	&pixel_stream,
				hls::stream<window>     	&window_stream);

		void Filter2D(
				hls::stream<window>         &window_stream,
				hls::stream<myDatatype>     &result_stream);
	};

}
