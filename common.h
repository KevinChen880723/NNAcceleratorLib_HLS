#define IMAGE_WIDTH 28
#define IMAGE_HEIGHT 28

#define FILTER_V_SIZE 3
#define FILTER_H_SIZE 3
#define layer1ChannelNum 1
#define layer2ChannelNum 10
#define layer3ChannelNum 8
#define layer4ChannelNum 6
#define layer5ChannelNum 4
#define FC1InfeatNum 64
#define FC1OutfeatNum 10

typedef float myDatatype;
typedef hls::stream<myDatatype> myStream;
