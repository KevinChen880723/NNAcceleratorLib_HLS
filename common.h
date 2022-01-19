#include "ap_fixed.h"
#include "hls_stream.h"

#define IMAGE_WIDTH 28
#define IMAGE_HEIGHT 28

// Kernel Information
#define FILTER_V_SIZE 3
#define FILTER_H_SIZE 3

// Channel of Each Layer
#define layer1ChannelNum 1
#define layer2ChannelNum 10
#define layer3ChannelNum 8
#define layer4ChannelNum 6
#define layer5ChannelNum 4

// Feature Map Size of Convolutional Layer


// Dimension of Each Fully Connected Layer
#define FC1InfeatNum 64
#define FC1OutfeatNum 10

//typedef ap_fixed<13, 8, AP_RND_ZERO, AP_SAT> myDatatype;
typedef float myDatatype;
typedef hls::stream<myDatatype> myStream;
