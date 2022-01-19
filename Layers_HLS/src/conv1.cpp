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

#include "../include/conv1.h"

#define PRINT
void ReadFromMem(
        unsigned short            width,
        unsigned short            height,
        const myDatatype                  weights[FILTER_V_SIZE*FILTER_H_SIZE],
        hls::stream<myDatatype>     &input_stream,
        hls::stream<myDatatype>     &coeff_stream,
        hls::stream<myDatatype>     &pixel_stream )
{
    unsigned short num_coefs = FILTER_V_SIZE*FILTER_H_SIZE;
    read_coefs: for (int i=0; i<num_coefs; i++) {
        myDatatype coef = weights[i];
        coeff_stream.write( coef );
    }

    read_image: for (int n = 0; n < height*width; n++) {
        myDatatype pix = input_stream.read();
		#ifdef PRINT
        	std::cout << "pix in ReadFromMem: " << pix << std::endl;
		#endif
        pixel_stream.write( pix );
    }
}

void Window2D(
        unsigned short          width,
        unsigned short          height,
        hls::stream<myDatatype>   &pixel_stream,
        hls::stream<window>     &window_stream,
        ap_int<1>               do_padding)
{
    // Line buffers - used to store [FILTER_V_SIZE-1] entire lines of pixels
    myDatatype LineBuffer[FILTER_V_SIZE-1][IMAGE_WIDTH];
#pragma HLS ARRAY_PARTITION variable=LineBuffer dim=1 complete
#pragma HLS DEPENDENCE variable=LineBuffer inter false
#pragma HLS DEPENDENCE variable=LineBuffer intra false

    // Sliding window of [FILTER_V_SIZE][FILTER_H_SIZE] pixels
    window Window;

    unsigned col_ptr = 0;
    // Initializing time to fill the data that raquired by a window into the line buffer
    // In the official code, because of the zero padding this is the time to get half the data in the first window
    unsigned ramp_up = (do_padding == 1) ? width*((FILTER_V_SIZE-1)/2)+(FILTER_H_SIZE-1)/2: width*(FILTER_V_SIZE-1)+(FILTER_H_SIZE-1);
    unsigned num_out_pixels = (do_padding == 1) ? width*height: width*height;//(width - FILTER_H_SIZE + 1)*(height - FILTER_V_SIZE + 1);
    unsigned num_iterations = num_out_pixels + ramp_up;

    // Iterate until all pixels have been processed
    update_window: for (int n=0; n<num_iterations; n++)
    {
#pragma HLS PIPELINE II=1

        // Read a new pixel from the input stream
        myDatatype new_pixel = (n<num_out_pixels) ? pixel_stream.read() : 0;
		#ifdef PRINT
        	std::cout << "new_pixel in Window2D: " << new_pixel << std::endl;
		#endif

        // Shift the window and add a column of new pixels from the line buffer
        for(int i = 0; i < FILTER_V_SIZE; i++) {
            for(int j = 0; j < FILTER_H_SIZE-1; j++) {
                Window.pix[i][j] = Window.pix[i][j+1];
            }
            Window.pix[i][FILTER_H_SIZE-1] = (i<FILTER_V_SIZE-1) ? LineBuffer[i][col_ptr] : new_pixel;
        }

        // Shift pixels in the column of pixels in the line buffer, add the newest pixel
        for(int i = 0; i < FILTER_V_SIZE-2; i++) {
            LineBuffer[i][col_ptr] = LineBuffer[i+1][col_ptr];
        }
        LineBuffer[FILTER_V_SIZE-2][col_ptr] = new_pixel;

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
        	if (col_idx < width - FILTER_H_SIZE + 1 && row_idx < height - FILTER_V_SIZE + 1)
        		window_stream.write(Window);
        	else{
        		if (do_padding == 1) window_stream.write(Window);
        	}
        }

    }
}

void Filter2D(
        unsigned short              width,
        unsigned short              height,
        hls::stream<myDatatype>       &coeff_stream,
        hls::stream<window>         &window_stream,
		hls::stream<myDatatype>       &output_stream,
        ap_int<1>                   do_padding)
{
    // Filtering coefficients
    myDatatype coeffs[FILTER_V_SIZE][FILTER_H_SIZE];
#pragma HLS ARRAY_PARTITION variable=coeffs complete dim=0

    // Load the coefficients into local storage
    load_coefs: for (int i=0; i<FILTER_V_SIZE; i++) {
        for (int j=0; j<FILTER_H_SIZE; j++) {
#pragma HLS PIPELINE II=1
            coeffs[i][j] = coeff_stream.read();
			#ifdef PRINT
            	std::cout << "coeffs[i][j] in Filter2D: " << coeffs[i][j] << std::endl;
			#endif
        }
    }

    // If we don't want to pad the image, the output size will be shinked
    // I only implemented stide=1, so the size after convolution will be (OriginalSize - WindowSize> + 1)
    height = (do_padding == true)? height: height-FILTER_V_SIZE+1;
    width = (do_padding == true)? width: width-FILTER_H_SIZE+1;

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
            for(int row=0; row<FILTER_V_SIZE; row++)
            {
                for(int col=0; col<FILTER_H_SIZE; col++)
                {
                    myDatatype pixel;
                    int xoffset = (x+col-(FILTER_H_SIZE/2));
                    int yoffset = (y+row-(FILTER_V_SIZE/2));
                    // Deal with boundary conditions : clamp pixels to 0 when outside of image
                    // Pad zero to the pixels that over the image
                    if ( do_padding==1 && ((xoffset<0) || (xoffset>=width) || (yoffset<0) || (yoffset>=height)) ) {
                        pixel = 0;
                    } else {
                        pixel = w.pix[row][col];
                    }
					#ifdef PRINT
            			std::cout << w.pix[row][col];
					#endif
            		if (col == FILTER_H_SIZE-1) std::cout << "\n" << std::endl;
            		else std::cout << "\t";

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
void Filter2DKernel(
        const myDatatype                 Wconv[FILTER_V_SIZE*FILTER_H_SIZE],
        unsigned short           width_input,
        unsigned short           height_input,
        hls::stream<myDatatype>    &input_stream,
        hls::stream<myDatatype>    &output_stream)
    {
#pragma HLS DATAFLOW

	// Stream of pixels from kernel input to filter, and from filter to output
    hls::stream<myDatatype,2>      coefs_stream("coefs_stream");
    hls::stream<myDatatype,2>      pixel_stream("pixel_stream");
    hls::stream<window,3>        window_stream("window_stream"); // Set FIFO depth to 0 to minimize resources

	// Read image data from global memory over AXI4 MM, and stream pixels out
    ReadFromMem(width_input, height_input, Wconv, input_stream, coefs_stream, pixel_stream);

    // Read incoming pixels and form valid HxV windows
    Window2D(width_input, height_input, pixel_stream, window_stream, 0);

	// Process incoming stream of pixels, and stream pixels out
	Filter2D(width_input, height_input, coefs_stream, window_stream, output_stream, 0);
    }

void summation(
        myDatatype                 bias,
        unsigned short           width,
        unsigned short           height,
        unsigned short           num_channel,
        hls::stream<myDatatype>    &ChannelOutput_stream,
        hls::stream<myDatatype>    &OverallOutput_stream)
{
    myDatatype outputFeature[height][width];
    for(int c = 0; c < num_channel; c++){
        for(int y = 0; y < height; y++){
            for(int x = 0; x < width; x++){
                if (c == 0) outputFeature[y][x] = ChannelOutput_stream.read();
                else{
                    outputFeature[y][x] += ChannelOutput_stream.read();
                }
                if (c == num_channel-1){
                    OverallOutput_stream.write(outputFeature[y][x] + bias);
                }
				#ifdef PRINT
                	std::cout << "outputFeature[" << y << "][" << x<< "] in summation: " << outputFeature[y][x] + bias << std::endl;
				#endif
            }
        }
    }
}

void pixelBuffer(
        unsigned short           width_input,
        unsigned short           height_input,
	    unsigned short 			 channel_input,
	    unsigned short 			 channel_output,
        hls::stream<myDatatype>    &input_stream,
        hls::stream<myDatatype>    &Buffer_stream)
{
    myDatatype inputBuffer[channel_input][height_input][width_input];
	for (int c_o = 0; c_o < channel_output; c_o++){
		for (int c_i = 0; c_i < channel_input; c_i++){
			for (int y = 0; y < height_input; y++){
				for (int x = 0; x < width_input; x++){
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


void conv1(
		const myDatatype                 Wconv[layer2ChannelNum][layer1ChannelNum][FILTER_V_SIZE*FILTER_H_SIZE],
		const myDatatype                 Bconv[layer2ChannelNum],
        unsigned short           width_input,
        unsigned short           height_input,
        hls::stream<myDatatype>    &input_stream,
        hls::stream<myDatatype>    &OverallOutput_stream)
{
//#pragma HLS interface ap_ctrl_none port=return
#pragma HLS DATAFLOW
	hls::stream<myDatatype> Buffer_stream("Buffer_stream");
    hls::stream<myDatatype> ChannelOutput_stream("ChannelOutput_stream");
    unsigned short width_output   = width_input - FILTER_H_SIZE + 1;
    unsigned short height_output  = height_input - FILTER_V_SIZE + 1;
    unsigned short channel_input  = layer1ChannelNum;
    unsigned short channel_output = layer2ChannelNum;

    pixelBuffer(width_input, height_input, channel_input, channel_output, input_stream, Buffer_stream);
    // Execute convolution <layer2CnannelNum> times to get the output with <layer2CnannelNum> channels
    for(int channel_num_o = 0; channel_num_o < channel_output; channel_num_o++){
        // Execute convolution for a kernel
        for (int channel_num_i = 0; channel_num_i < channel_input; channel_num_i++){
            // Do convolution on every channels, then send the output stream to summation module
			#ifdef PRINT
				std::cout << "================================================================================================" << std::endl;
				std::cout << "========================================= kernel: " << channel_num_o << " ============================================" << std::endl;
				std::cout << "================================================================================================" << std::endl;
			#endif
            Filter2DKernel(Wconv[channel_num_o][channel_num_i], width_input, height_input, Buffer_stream, ChannelOutput_stream);
        }
        // Summation module sum the value in the same coordinate up then add by the bias
        summation(Bconv[channel_num_o], width_output, height_output, channel_input, ChannelOutput_stream, OverallOutput_stream);
    }
}

/*
 * ??‘ä?å¤ªç¢ºå?šæ?å¾Œä??‹Moduleä¸­ç?„summation()è¦æ”¾?œ¨??žå?ˆå…§??„æ˜¯å¤–ï?Œæ”¾?œ¨è£¡é¢??‘æ?•æ?ƒæ?‰ä?‰å?‹ä?æ¨???„ç¡¬é«? (??‘è?ä?‰æ¬¡Filter2DKernel()?ƒ½å°æ?‰åˆ°??Œä??‹summation())ï¼Œæ”¾å¤–é¢ä¸çŸ¥??“æ?ƒä?æ?ƒç?‰è¿´??ˆå…§å®¹ç?æ?Ÿæ?åŸ·è¡??
 * ??Ÿè¦º?”¾è£¡é¢??‰è©²ä¸æ?ƒæ?‰ä?‰å?‹ä?æ¨???„ç¡¬é«”ï?Œå? ç‚º??‘æ?’æ?‰Unroll?‚å?‚æ?œå?ƒè?Šæ?ä?‰å?‹ç¡¬é«”å?Œæ­¥?Ÿ·è¡Œï?Œæ?‘å°±ä¸æ?ƒçŸ¥??“ä?–åŸ·è¡Œç?„é?†å?æ˜¯?Žéº¼æ¨?ï¼Œsummation()è£¡é¢??è¨­ä¸?å±¤ä?å±¤è?‘ç?„é?†å?å°±ä¸ä?å®šå?ä?†ã??
 */
