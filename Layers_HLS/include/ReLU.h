#include "hls_stream.h"
#include "../../common.h"

#ifndef RELU_H
#define RELU_H

namespace YKHLS{
	class ReLU{
	public:
		ReLU();
		ReLU(   unsigned short c,
				unsigned short w,
				unsigned short h);
		void operator()(
				hls::stream<myDatatype>    	&input_stream,
				hls::stream<myDatatype>    	&output_stream);
	private:
		unsigned short channel_num;
		unsigned short width;
		unsigned short height;
	};
}
#endif
