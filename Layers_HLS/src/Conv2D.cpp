/*
* Copyright 2021 Xilinx, Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "../include/Conv2D.h"
#include "hls_stream.h"

#ifndef __SYNTHESIS__
//	#define PRINT
#endif

namespace YKHLS{
	template<	unsigned short 				width_filter,
				unsigned short 				height_filter,
				unsigned short 				channel_input,
				unsigned short 				channel_output,
				unsigned short 				width_input,
				unsigned short 				height_input,
				unsigned short 				width_output,
				unsigned short 				height_output>
	Conv2D<width_filter, height_filter, channel_input, channel_output, width_input, height_input, width_output, height_output>::Conv2D(){}

	template<	unsigned short 				width_filter,
				unsigned short 				height_filter,
				unsigned short 				channel_input,
				unsigned short 				channel_output,
				unsigned short 				width_input,
				unsigned short 				height_input,
				unsigned short 				width_output,
				unsigned short 				height_output>
	void Conv2D<width_filter, height_filter, channel_input, channel_output, width_input, height_input, width_output, height_output>::ReadFromMem(
				unsigned short            	width,
				unsigned short            	height,
				const myDatatype            weights[height_filter*width_filter],
				hls::stream<myDatatype>     &input_stream,
				hls::stream<myDatatype>     &coeff_stream,
				hls::stream<myDatatype>     &pixel_stream )
	{
#pragma HLS interface ap_ctrl_none port=return
		unsigned short num_coefs = height_filter*width_filter;
		read_coefs: for (int i=0; i<num_coefs; i++) {
			myDatatype coef = weights[i];
			coeff_stream.write( coef );
		}

		read_image: for (int n = 0; n < height*width; n++) {
#pragma HLS PIPELINE II=1
			myDatatype pix = input_stream.read();
			#ifdef PRINT
				std::cout << "pix in ReadFromMem: " << pix << std::endl;
			#endif
			pixel_stream.write( pix );
		}
	}

	template<	unsigned short 				width_filter,
				unsigned short 				height_filter,
				unsigned short 				channel_input,
				unsigned short 				channel_output,
				unsigned short 				width_input,
				unsigned short 				height_input,
				unsigned short 				width_output,
				unsigned short 				height_output>
	void Conv2D<width_filter, height_filter, channel_input, channel_output, width_input, height_input, width_output, height_output>::Window2D(
			unsigned short          	width,
			unsigned short          	height,
			hls::stream<myDatatype>   	&pixel_stream,
			hls::stream<window>     	&window_stream,
			ap_int<1>               	do_padding)
	{
#pragma HLS interface ap_ctrl_none port=return
		// Line buffers - used to store [height_filter-1] entire lines of pixels
		myDatatype LineBuffer[height_filter-1][width_input];
		// Use this pragma to partition the physical memory of LineBuffer into many separate memory blocks.
	#pragma HLS ARRAY_PARTITION variable=LineBuffer dim=1 complete
	#pragma HLS DEPENDENCE variable=LineBuffer inter false
	#pragma HLS DEPENDENCE variable=LineBuffer intra false

		// Sliding window of [height_filter][width_filter] pixels
		window Window;

		unsigned col_ptr = 0;
		// Initializing time to fill the data that raquired by a window into the line buffer
		// In the official code, because of the zero padding this is the time to get half the data in the first window
		unsigned ramp_up = (do_padding == 1) ? width*((height_filter-1)/2)+(width_filter-1)/2: width*(height_filter-1)+(width_filter-1);
		unsigned num_pixels = width*height;
		unsigned num_iterations = num_pixels + ramp_up;

		// Iterate until all pixels have been processed
		update_window: for (int n=0; n<num_iterations; n++)
		{
	#pragma HLS PIPELINE II=1

			// Read a new pixel from the input stream
			myDatatype new_pixel = (n<num_pixels) ? pixel_stream.read() : myDatatype(0);
			#ifdef PRINT
				std::cout << "new_pixel in Window2D: " << new_pixel << std::endl;
			#endif

			// Shift the window and add a column of new pixels from the line buffer
			for(int i = 0; i < height_filter; i++) {
				for(int j = 0; j < width_filter-1; j++) {
					Window.pix[i][j] = Window.pix[i][j+1];
				}
				Window.pix[i][width_filter-1] = (i<height_filter-1) ? LineBuffer[i][col_ptr] : new_pixel;
			}

			// Shift pixels in the column of pixels in the line buffer, add the newest pixel
			for(int i = 0; i < height_filter-2; i++) {
				LineBuffer[i][col_ptr] = LineBuffer[i+1][col_ptr];
			}
			LineBuffer[height_filter-2][col_ptr] = new_pixel;

			// Update the line buffer column pointer
			if (col_ptr==(width - 1)) {
				col_ptr = 0;
			} else {
				col_ptr++;
			}

			// Write output only when enough pixels have been read the buffers and ramped-up
			if (n>=ramp_up) {
				unsigned short col_idx = (n - ramp_up) % width;
				unsigned short row_idx = (n - ramp_up) / height;
				if (col_idx < width - width_filter + 1 && row_idx < height - height_filter + 1)
					window_stream.write(Window);
				else{
					if (do_padding == 1) window_stream.write(Window);
				}
			}

		}
	}

	template<	unsigned short 				width_filter,
				unsigned short 				height_filter,
				unsigned short 				channel_input,
				unsigned short 				channel_output,
				unsigned short 				width_input,
				unsigned short 				height_input,
				unsigned short 				width_output,
				unsigned short 				height_output>
	void Conv2D<width_filter, height_filter, channel_input, channel_output, width_input, height_input, width_output, height_output>::Filter2D(
			unsigned short              width,
			unsigned short              height,
			hls::stream<myDatatype>     &coeff_stream,
			hls::stream<window>         &window_stream,
			hls::stream<myDatatype>     &output_stream,
			ap_int<1>                   do_padding)
	{
#pragma HLS interface ap_ctrl_none port=return
		// Filtering coefficients
		myDatatype coeffs[height_filter][width_filter];
	#pragma HLS ARRAY_PARTITION variable=coeffs complete dim=0

		// Load the coefficients into local storage
		load_coefs: for (int i=0; i<height_filter; i++) {
			for (int j=0; j<width_filter; j++) {
	#pragma HLS PIPELINE II=1
				coeffs[i][j] = coeff_stream.read();
				#ifdef PRINT
					std::cout << "coeffs[i][j] in Filter2D: " << coeffs[i][j] << std::endl;
				#endif
			}
		}

		// If we don't want to pad the image, the output size will be shinked
		// I only implemented stide=1, so the size after convolution will be (OriginalSize - WindowSize> + 1)
		height = (do_padding == true)? height: height-height_filter+1;
		width = (do_padding == true)? width: width-width_filter+1;

		// Process the incoming stream of pixel windows
		apply_filter: for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
	#pragma HLS PIPELINE II=1
				// Read a 2D window of pixels
				window w = window_stream.read();
				#ifdef PRINT
					std::cout << "window in Filter2D(" << y << ", " << x << ")" << std::endl;
				#endif
				// Apply filter to the 2D window
				myDatatype sum = 0.0;
				for(int row=0; row<height_filter; row++)
				{
					for(int col=0; col<width_filter; col++)
					{
						myDatatype pixel;
						int xoffset = (x+col-(width_filter/2));
						int yoffset = (y+row-(height_filter/2));
						// Deal with boundary conditions : clamp pixels to 0 when outside of image
						// Pad zero to the pixels that over the image
						if ( do_padding==1 && ((xoffset<0) || (xoffset>=width) || (yoffset<0) || (yoffset>=height)) ) {
							pixel = 0;
						} else {
							pixel = w.pix[row][col];
						}
						#ifdef PRINT
							std::cout << w.pix[row][col];
							if (col == width_filter-1) std::cout << "\n" << std::endl;
							else std::cout << "\t";
						#endif

						sum += pixel*(myDatatype)coeffs[row][col];
					}
				}
				// Write the output pixel
				output_stream.write(sum);
				#ifdef PRINT
					std::cout << "------------------------------------------------------------------" << std::endl;
					std::cout << "Sum in the window = " << sum << "\t<<" << y*width+x << ">>" << std::endl;
					std::cout << "------------------------------------------------------------------" << std::endl;
				#endif
			}
		}
	}


	/*
	 * A module used for executing one channel convolution
	 */
	template<	unsigned short 				width_filter,
				unsigned short 				height_filter,
				unsigned short 				channel_input,
				unsigned short 				channel_output,
				unsigned short 				width_input,
				unsigned short 				height_input,
				unsigned short 				width_output,
				unsigned short 				height_output>
	void Conv2D<width_filter, height_filter, channel_input, channel_output, width_input, height_input, width_output, height_output>::Filter2DKernel(
			const myDatatype            Wconv[height_filter*width_filter],
			hls::stream<myDatatype>     &input_stream,
			hls::stream<myDatatype>     &output_stream)
		{
#pragma HLS interface ap_ctrl_none port=return
	#pragma HLS DATAFLOW

		// Stream of pixels from kernel input to filter, and from filter to output
		hls::stream<myDatatype,10>      coefs_stream("coefs_stream");
		hls::stream<myDatatype,10>      pixel_stream("pixel_stream");
		hls::stream<window,10>        window_stream("window_stream"); // Set FIFO depth to 0 to minimize resources

		// Read image data from global memory over AXI4 MM, and stream pixels out
		ReadFromMem(width_input, height_input, Wconv, input_stream, coefs_stream, pixel_stream);

		// Read incoming pixels and form valid HxV windows
		Window2D(width_input, height_input, pixel_stream, window_stream, 0);

		// Process incoming stream of pixels, and stream pixels out
		Filter2D(width_input, height_input, coefs_stream, window_stream, output_stream, 0);
		}

	template<	unsigned short 				width_filter,
				unsigned short 				height_filter,
				unsigned short 				channel_input,
				unsigned short 				channel_output,
				unsigned short 				width_input,
				unsigned short 				height_input,
				unsigned short 				width_output,
				unsigned short 				height_output>
	void Conv2D<width_filter, height_filter, channel_input, channel_output, width_input, height_input, width_output, height_output>::summation(
			myDatatype 					outputFeature[height_output][width_output],
			const myDatatype            bias,
			hls::stream<myDatatype>     &ChannelOutput_stream,
			hls::stream<myDatatype>     &OverallOutput_stream)
	{
#pragma HLS interface ap_ctrl_none port=return
		for(int c = 0; c < channel_input; c++){
			for(int y = 0; y < height_output; y++){
				for(int x = 0; x < width_output; x++){
		#pragma HLS PIPELINE II=1
					myDatatype pix = ChannelOutput_stream.read();
					if (c == 0) outputFeature[y][x] = pix;
					else{
						outputFeature[y][x] += pix;
					}
					if (c == channel_input-1){
						OverallOutput_stream.write(outputFeature[y][x] + bias);
					}
					#ifdef PRINT
						std::cout << "outputFeature[" << y << "][" << x<< "] in summation: " << outputFeature[y][x] + bias << std::endl;
					#endif
				}
			}
		}
	}

	template<	unsigned short 				width_filter,
				unsigned short 				height_filter,
				unsigned short 				channel_input,
				unsigned short 				channel_output,
				unsigned short 				width_input,
				unsigned short 				height_input,
				unsigned short 				width_output,
				unsigned short 				height_output>
	void Conv2D<width_filter, height_filter, channel_input, channel_output, width_input, height_input, width_output, height_output>::pixelBuffer(
			hls::stream<myDatatype>     &input_stream,
			hls::stream<myDatatype>     &Buffer_stream)
	{
#pragma HLS interface ap_ctrl_none port=return
		const unsigned short channel_input_const = channel_input;
		const unsigned short height_input_const = height_input;
		const unsigned short width_input_const = width_input;
		myDatatype inputBuffer[channel_input_const][height_input_const][width_input_const];
		for (int c_o = 0; c_o < channel_output; c_o++){
			for (int c_i = 0; c_i < channel_input; c_i++){
				for (int y = 0; y < height_input; y++){
					for (int x = 0; x < width_input; x++){
#pragma HLS PIPELINE II=1
						if (c_o == 0){
							myDatatype temp = input_stream.read();
							Buffer_stream.write(temp);
							inputBuffer[c_i][y][x] = temp;
						}
						else Buffer_stream.write(inputBuffer[c_i][y][x]);
					}
				}
			}
		}
	}

	template<	unsigned short 				width_filter,
				unsigned short 				height_filter,
				unsigned short 				channel_input,
				unsigned short 				channel_output,
				unsigned short 				width_input,
				unsigned short 				height_input,
				unsigned short 				width_output,
				unsigned short 				height_output>
	void Conv2D<width_filter, height_filter, channel_input, channel_output, width_input, height_input, width_output, height_output>::ConvolutionWithoutSum(
			const myDatatype                 Kernel[channel_input][height_filter*width_filter],
			hls::stream<myDatatype>    		 &Buffer_stream,
			hls::stream<myDatatype>    		 &ChannelOutput_stream)
	{
		for (int channel_num_i = 0; channel_num_i < channel_input; channel_num_i++){
			// Do convolution on every channels, then send the output stream to summation module
			Filter2DKernel(Kernel[channel_num_i], Buffer_stream, ChannelOutput_stream);
		}
	}

	template<	unsigned short 				width_filter,
				unsigned short 				height_filter,
				unsigned short 				channel_input,
				unsigned short 				channel_output,
				unsigned short 				width_input,
				unsigned short 				height_input,
				unsigned short 				width_output,
				unsigned short 				height_output>
	void Conv2D<width_filter, height_filter, channel_input, channel_output, width_input, height_input, width_output, height_output>::ExecuteConv(
			const myDatatype                 Wconv[channel_output][channel_input][height_filter*width_filter],
			const myDatatype                 Bconv[channel_output],
			hls::stream<myDatatype>    		 &Buffer_stream,
			hls::stream<myDatatype>    		 &OverallOutput_stream)
	{
#pragma HLS interface ap_ctrl_none port=return
		hls::stream<myDatatype, channel_input*channel_output*width_input*height_input> ChannelOutput_stream("ChannelOutput_stream");
		myDatatype sumBuffer[height_output][width_output];
		// Execute convolution <layer2CnannelNum> times to get the output with <layer2CnannelNum> channels
		for(int channel_num_o = 0; channel_num_o < channel_output; channel_num_o++){
			ConvolutionWithoutSum(Wconv[channel_num_o], Buffer_stream, ChannelOutput_stream);
			summation(sumBuffer, Bconv[channel_num_o], ChannelOutput_stream, OverallOutput_stream);
		}

	}

	template<	unsigned short 				width_filter,
				unsigned short 				height_filter,
				unsigned short 				channel_input,
				unsigned short 				channel_output,
				unsigned short 				width_input,
				unsigned short 				height_input,
				unsigned short 				width_output,
				unsigned short 				height_output>
	void Conv2D<width_filter, height_filter, channel_input, channel_output, width_input, height_input, width_output, height_output>::operator()(
			const myDatatype                 Wconv[channel_output][channel_input][height_filter*width_filter],
			const myDatatype                 Bconv[channel_output],
			hls::stream<myDatatype>    		 &input_stream,
			hls::stream<myDatatype>    		 &OverallOutput_stream)
	{
#pragma HLS interface ap_ctrl_none port=return
	#pragma HLS DATAFLOW

		hls::stream<myDatatype, 10> Buffer_stream("Buffer_stream");

		pixelBuffer(input_stream, Buffer_stream);
		ExecuteConv(Wconv, Bconv, Buffer_stream, OverallOutput_stream);
	}
}
