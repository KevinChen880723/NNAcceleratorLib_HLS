#include "../include/MaxPool2D.h"

namespace YKHLS{

	#ifndef __SYNTHESIS__
//		#define PRINT
	#endif

	template<	unsigned short 	width_filter,
				unsigned short 	height_filter>
	MaxPool2D<width_filter, height_filter>::MaxPool2D(): height_input(1), width_input(1), channel_input(1){}

	template<	unsigned short 	width_filter,
				unsigned short 	height_filter>
	MaxPool2D<width_filter, height_filter>::MaxPool2D(
			unsigned short w_i,
			unsigned short h_i,
			unsigned short c_i)
	{
		width_input = w_i;
		height_input = h_i;
		channel_input = c_i;
	}

	template<	unsigned short 	width_filter,
				unsigned short 	height_filter>
	void MaxPool2D<width_filter, height_filter>::ReadFromMem(
					hls::stream<myDatatype>     &input_stream,
					hls::stream<myDatatype>     &pixel_stream)
	{
	#pragma HLS interface ap_ctrl_none port=return
		std::cout << "\n========================================== ReadFromMem =================================================" << std::endl;
		std::cout << "width_filter = " << width_filter << std::endl;
		std::cout << "height_filter = " << height_filter << std::endl;
		std::cout << "width_input = " << width_input << std::endl;
		std::cout << "height_input = " << height_input << std::endl;
		std::cout << "channel_input = " << channel_input << std::endl;
		std::cout << "==========================================================================================================" << std::endl;
		for (int i = 0; i < channel_input*height_input*width_input; i++){
			myDatatype pix = input_stream.read();
			#ifdef PRINT
				std::cout << "pix in ReadFromMem (MaxPool2D): " << pix << std::endl;
			#endif
			pixel_stream.write(pix);
		}
	}

	template<	unsigned short 	width_filter,
				unsigned short 	height_filter>
	void MaxPool2D<width_filter, height_filter>::Window2D(
					hls::stream<myDatatype>   	&pixel_stream,
					hls::stream<window>     	&window_stream)
	{
	#pragma HLS interface ap_ctrl_none port=return
		std::cout << "\n========================================== Window2D =================================================" << std::endl;
		std::cout << "width_filter = " << width_filter << std::endl;
		std::cout << "height_filter = " << height_filter << std::endl;
		std::cout << "width_input = " << width_input << std::endl;
		std::cout << "height_input = " << height_input << std::endl;
		std::cout << "channel_input = " << channel_input << std::endl;
		std::cout << "==========================================================================================================" << std::endl;
		const unsigned short width_filter_const = width_filter;
		const unsigned short height_filter_const = height_filter;
		myDatatype LineBuffer[height_filter_const][width_filter_const];
		unsigned short row = 0;
		unsigned short col = 0;
		window Window;

		for (int c = 0; c < channel_input; c++){
			for (int y = 0; y < height_input; y++){
				for (int x = 0; x < width_input; x++){
					LineBuffer[row][x] = pixel_stream.read();
					#ifdef PRINT
						std::cout << "LineBuffer[" << row << "][" << x << "] in Window2D (MaxPool2D): " << LineBuffer[row][x] << std::endl;
					#endif
					if (row == height_filter-1 && col == width_filter-1){
						// load the corresponding pixels in LineBuffer into Window
						for (int h_f = 0; h_f < height_filter; h_f++){
							for (int w_f = 0; w_f < width_filter; w_f++){
								Window.pix[h_f][w_f] = LineBuffer[row-(height_filter-1)+h_f][x-(width_filter-1)+w_f];
							}
						}
						// Write Window into hls::stream
						window_stream.write(Window);
					}
					if (col == width_filter-1 || x == width_input-1) col = 0;
					else col++;
				}
				if (row == height_filter-1) row = 0;
				else row++;
			}
		}

	}

	template<	unsigned short 	width_filter,
				unsigned short 	height_filter>
	void MaxPool2D<width_filter, height_filter>::Filter2D(
				hls::stream<window>         &window_stream,
				hls::stream<myDatatype>     &result_stream)
	{
	#pragma HLS interface ap_ctrl_none port=return
		std::cout << "\n========================================== Filter2D =================================================" << std::endl;
		std::cout << "width_filter = " << width_filter << std::endl;
		std::cout << "height_filter = " << height_filter << std::endl;
		std::cout << "width_input = " << width_input << std::endl;
		std::cout << "height_input = " << height_input << std::endl;
		std::cout << "channel_input = " << channel_input << std::endl;
		std::cout << "==========================================================================================================" << std::endl;
		for (int c = 0; c < channel_input; c++){
			for (int y = 0; y < height_input/height_filter; y++){
				for (int x = 0; x < width_input/width_filter; x++){
					myDatatype max = 0;
					window w = window_stream.read();
					#ifdef PRINT
						std::cout << "w in Filter2D(" << y << ", " << x << ") (MaxPool2D)" << std::endl;
					#endif
					// Run over the window then get the maximum value within the window
					for (int row = 0; row < height_filter; row++){
						for (int col = 0; col < width_filter; col++){
							if (row == 0 && col == 0) max = w.pix[row][col];
							else{
								if (w.pix[row][col] > max)
									max = w.pix[row][col];
							}

							#ifdef PRINT
								std::cout << w.pix[row][col];
								if (col == width_filter-1) std::cout << "\n" << std::endl;
								else std::cout << "\t";
							#endif

						}
					}
					#ifdef PRINT
						std::cout << "------------------------------------------------------------------" << std::endl;
						std::cout << "max in the window = " << max << std::endl;
						std::cout << "------------------------------------------------------------------" << std::endl;
					#endif
					result_stream.write(max);

				}
			}
		}
	}

	template<	unsigned short 	width_filter,
				unsigned short 	height_filter>
	void MaxPool2D<width_filter, height_filter>::operator()(
				hls::stream<myDatatype>    	&input_stream,
				hls::stream<myDatatype>     &output_stream)
	{
	#pragma HLS interface ap_ctrl_none port=return
	#pragma HLS dataflow
		std::cout << "\n========================================== operator() =================================================" << std::endl;
		std::cout << "width_filter = " << width_filter << std::endl;
		std::cout << "height_filter = " << height_filter << std::endl;
		std::cout << "width_input = " << width_input << std::endl;
		std::cout << "height_input = " << height_input << std::endl;
		std::cout << "channel_input = " << channel_input << std::endl;
		std::cout << "==========================================================================================================" << std::endl;
		hls::stream<myDatatype> pixel_stream("pixel_stream");
		hls::stream<window>     window_stream("window_stream");
		ReadFromMem(input_stream, pixel_stream);
		Window2D(pixel_stream, window_stream);
		Filter2D(window_stream, output_stream);
	}
}
