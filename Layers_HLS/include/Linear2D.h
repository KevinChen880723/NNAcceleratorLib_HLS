#include "ap_int.h"
#include "hls_stream.h"
#include "../../common.h"

#ifndef LINEAR2D_H
#define LINEAR2D_H

namespace YKHLS{
	template<	unsigned short 				dim_output,
				unsigned short 				dim_input>
	class Linear2D{
	public:
		Linear2D();
		void operator()(
				const myDatatype            Wconv[dim_output][dim_input],
				const myDatatype            Bconv[dim_output],
				hls::stream<myDatatype>    	&input_stream,
				hls::stream<myDatatype>    	&output_stream);
	};
}

#endif
