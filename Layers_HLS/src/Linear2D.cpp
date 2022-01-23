#include "../include/Linear2D.h"
namespace YKHLS{
	template< unsigned short dim_output,
			  unsigned short dim_input>
	Linear2D<dim_output, dim_input>::Linear2D(){}

	template< unsigned short dim_output,
			  unsigned short dim_input>
	void Linear2D<dim_output, dim_input>::operator()(
			const myDatatype            Wconv[dim_output][dim_input],
			const myDatatype            Bconv[dim_output],
			hls::stream<myDatatype>    	&input_stream,
			hls::stream<myDatatype>    	&output_stream)
	{
#pragma HLS interface ap_ctrl_none port=return
		myDatatype feature[dim_input];
//		#pragma HLS ARRAY_PARTITION variable=feature dim=0 block factor=8

		for (int i = 0; i < dim_input; i++){
			feature[i] = input_stream.read();
		}
		for (int col = 0; col < dim_output; col++){
//		#pragma HLS PIPELINE II=1
			myDatatype sum = 0;
			for (int row = 0; row < dim_input; row++){
				if (row == 0) sum = Wconv[col][row] * feature[row];
				else sum += Wconv[col][row] * feature[row];
			}
			sum += Bconv[col];
			output_stream.write(sum);
		}
	}
}
