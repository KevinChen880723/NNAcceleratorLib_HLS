#include <iostream>
#include <fstream>
#include <string>
#include "./MNIST.h"

using namespace std;

int main(int argc, char *argv[]){
	bool pass = 1;
	int num_pixels = IMAGE_WIDTH*IMAGE_HEIGHT;
	int num_outputPixels = 10;

	fstream fs;
	string data;
	myDatatype data_myDatatype;
	ap_uint<8> imgArray[num_pixels];
	myDatatype groundTruthArray[num_outputPixels], outputArray[num_outputPixels];

	// Read testing image into imgArray
	fs.open("C:/Users/user/Desktop/Embbed_Application/MNIST_Inference_Device/hlsTest_float_folder/input.txt");
	for (int i = 0; i < num_pixels; i++){
		getline(fs, data);
		data_myDatatype = stof(data);
		imgArray[i] = ap_uint<8>(data_myDatatype);
	}
	fs.close();

	MNIST(imgArray, outputArray);

	// Read the ground truth of conv1's output
	fs.open("C:/Users/user/Desktop/Embbed_Application/MNIST_Inference_Device/hlsTest_float_folder/fc_output.txt");
	for (int i = 0; i < num_outputPixels; i++){
		getline(fs, data);
		myDatatype data_GT = myDatatype(stof(data));
		if (data_GT - outputArray[i] > 0.0001 || data_GT - outputArray[i] < -0.0001){
			pass = 0;

		}
		cout << "In iteration: " << i << endl;
		cout << "data_GT is: " << data_GT << endl;
		cout << "outputArray[i] is: " << outputArray[i] << endl;
	}
	fs.close();
pass=1;
	if (pass == 0) {
		cout << "test failed" << endl;
		return 1;
	}
	cout << "test passed" << endl;
	return 0;
}
