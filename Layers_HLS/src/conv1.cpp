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

#include "conv1.h"

void ReadFromMem(
        unsigned short            width,
        unsigned short            height,
        datatype                  *weights,
        hls::stream<datatype>     &input_stream,
        hls::stream<datatype>     &coeff_stream,
        hls::stream<datatype>     &pixel_stream )
{
    assert(height <= MAX_IMAGE_HEIGHT);

    unsigned short num_coefs = FILTER_V_SIZE*FILTER_H_SIZE;
    read_coefs: for (int i=0; i<num_coefs; i++) {
        datatype coef = weights[i];
        coeff_stream.write( coef );
    }

    read_image: for (int n = 0; n < height*width; n++) {
        datatype pix = input_stream.read();
        pixel_stream.write( pix );
    }
}

struct window {
    datatype pix[FILTER_V_SIZE][FILTER_H_SIZE];
};

void Window2D(
        unsigned short          width,
        unsigned short          height,
        hls::stream<datatype>   &pixel_stream,
        hls::stream<window>     &window_stream,
        ap_int<1>               do_padding)
{
    // Line buffers - used to store [FILTER_V_SIZE-1] entire lines of pixels
    datatype LineBuffer[FILTER_V_SIZE-1][MAX_IMAGE_WIDTH];
#pragma HLS ARRAY_PARTITION variable=LineBuffer dim=1 complete
#pragma HLS DEPENDENCE variable=LineBuffer inter false
#pragma HLS DEPENDENCE variable=LineBuffer intra false

    // Sliding window of [FILTER_V_SIZE][FILTER_H_SIZE] pixels
    window Window;

    unsigned col_ptr = 0;
    // Initializing time to fill the data that raquired by a window into the line buffer
    // In the official code, because of the zero padding this is the time to get half the data in the first window
    unsigned ramp_up = (do_padding == 1) ? width*((FILTER_V_SIZE-1)/2)+(FILTER_H_SIZE-1)/2: width*(FILTER_V_SIZE-1)+(FILTER_H_SIZE-1);
    unsigned num_pixels = width*height;
    unsigned num_iterations = num_pixels + ramp_up;

    const unsigned max_iterations = MAX_IMAGE_WIDTH*MAX_IMAGE_HEIGHT + MAX_IMAGE_WIDTH*((FILTER_V_SIZE-1)/2)+(FILTER_H_SIZE-1)/2;

    // Iterate until all pixels have been processed
    update_window: for (int n=0; n<num_iterations; n++)
    {
#pragma HLS LOOP_TRIPCOUNT max=max_iterations
#pragma HLS PIPELINE II=1

        // Read a new pixel from the input stream
        datatype new_pixel = (n<num_pixels) ? pixel_stream.read() : 0;

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
        if (col_ptr==(width-1)) {
            col_ptr = 0;
        } else {
            col_ptr++;
        }

        // Write output only when enough pixels have been read the buffers and ramped-up
        if (n>=ramp_up) {
            window_stream.write(Window);
        }

    }
}

void Filter2D(
        unsigned short              width,
        unsigned short              height,
        hls::stream<datatype>       &coeff_stream,
        hls::stream<window>         &window_stream,
		hls::stream<datatype>       &output_stream,
        ap_int<1>                   do_padding)
{
    assert(width  <= MAX_IMAGE_WIDTH);
    assert(height <= MAX_IMAGE_HEIGHT);

    // Filtering coefficients
    datatype coeffs[FILTER_V_SIZE][FILTER_H_SIZE];
#pragma HLS ARRAY_PARTITION variable=coeffs complete dim=0

    // Load the coefficients into local storage
    load_coefs: for (int i=0; i<FILTER_V_SIZE; i++) {
        for (int j=0; j<FILTER_H_SIZE; j++) {
#pragma HLS PIPELINE II=1
            coeffs[i][j] = coeff_stream.read();
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

            // Apply filter to the 2D window
            int sum = 0;
            for(int row=0; row<FILTER_V_SIZE; row++)
            {
                for(int col=0; col<FILTER_H_SIZE; col++)
                {
                    datatype pixel;
                    int xoffset = (x+col-(FILTER_H_SIZE/2));
                    int yoffset = (y+row-(FILTER_V_SIZE/2));
                    // Deal with boundary conditions : clamp pixels to 0 when outside of image
                    // Pad zero to the pixels that over the image
                    if ( do_padding==1 && ((xoffset<0) || (xoffset>=width) || (yoffset<0) || (yoffset>=height)) ) {
                        pixel = 0;
                    } else {
                        pixel = w.pix[row][col];
                    }
                    sum += pixel*(datatype)coeffs[row][col];
                }
            } 
            // Write the output pixel
            output_stream.write(sum);
        }
    }
}


extern "C" {

/*
 * A module used for executing one channel convolution
 */
void Filter2DKernel(
        datatype                 *Wconv,
        unsigned short           width_input,
        unsigned short           height_input,
        hls::stream<datatype>    &input_stream,
        hls::stream<datatype>    &output_stream)
    {
#pragma HLS DATAFLOW

	// Stream of pixels from kernel input to filter, and from filter to output
    hls::stream<datatype,2>      coefs_stream;
    hls::stream<datatype,2>      pixel_stream;
    hls::stream<window,3>        window_stream; // Set FIFO depth to 0 to minimize resources
    // hls::stream<datatype,64>     output_stream;

	// Read image data from global memory over AXI4 MM, and stream pixels out
    ReadFromMem(width_input, height_input, Wconv, coefs_stream, input_stream, pixel_stream);

    // Read incoming pixels and form valid HxV windows
    Window2D(width_input, height_input, pixel_stream, window_stream, 0);

	// Process incoming stream of pixels, and stream pixels out
	Filter2D(width_input, height_input, coefs_stream, window_stream, output_stream, 0);

	// // Write incoming stream of pixels and write them to global memory over AXI4 MM
	// WriteToMem(width_input, height_input, output_stream, dst);

    }

}

void summation(
        datatype                 bias,
        unsigned short           width,
        unsigned short           height,
        unsigned short           num_channel,
        hls::stream<datatype>    &ChannelOutput_stream,
        hls::stream<datatype>    &OverallOutput_stream)
{
    datatype outputFeature[height][width];
    for(int c = 0; c < num_channel; c++){
        for(int y = 0; y < height; y++){
            for(int x = 0; x < width; x++){
                if (c == 0) outputFeature[y][x] = ChannelOutput_stream.read();
                else{
                    outputFeature[y][x] += ChannelOutput_stream.read();
                    if (c == num_channel-1){
                        OverallOutput_stream.write(outputFeature[y][x] + bias);
                    }
                }
            }
        }
    }
}

void conv1(
        datatype                 Wconv[layer2CnannelNum][layer1CnannelNum][FILTER_V_SIZE*FILTER_H_SIZE],
        datatype                 Bconv[layer2CnannelNum],
        unsigned short           width_input,
        unsigned short           height_input,
        hls::stream<datatype>    &input_stream,
        hls::stream<datatype>    &OverallOutput_stream)
{
    hls::stream<datatype> ChannelOutput_stream;
    unsigned short width_output   = width_input - FILTER_H_SIZE + 1;
    unsigned short height_output  = height_input - FILTER_V_SIZE + 1;
    unsigned short channel_input  = layer1CnannelNum;
    unsigned short channel_output = layer2CnannelNum;
    datatype outputFeature[height][width];
    
    // Execute convolution <layer2CnannelNum> times to get the output with <layer2CnannelNum> channels
    for(int channel_num_o = 0; channel_num_o < channel_output; channel_num_o++)){
        // Execute convolution for a kernel
        for (int channel_num_i = 0; channel_num_i < channel_input; channel_num_i++){
            // Do convolution on every channels, then send the output stream to summation module
            Filter2DKernel(Wconv[channel_num_o][channel_num_i], width_input, height_input, input_stream, ChannelOutput_stream);
        }
        // Summation module sum the value in the same coordinate up then add by the bias
        summation(Bconv[channel_num_o], width_input, height_input, channel_input, ChannelOutput_stream, OverallOutput_stream);
    }
}

/*
 * 我不太確定最後一個Module中的summation()要放在回圈內還是外，放在裡面我怕會有三個一樣的硬體 (我要三次Filter2DKernel()都對應到同一個summation())，放外面不知道會不會等迴圈內容結束才執行?
 * 感覺放裡面應該不會有三個一樣的硬體，因為我沒有Unroll。如果它變成三個硬體同步執行，我就不會知道他執行的順序是怎麼樣，summation()裡面預設一層一層跑的順序就不一定對了。
 */