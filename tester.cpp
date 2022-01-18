#include <iostream>
#include <fstream>
#include <string>
#include "./MNIST.h"

using namespace std;

int main(int argc, char *argv[]){
	bool pass = 1;
	int num_pixels = IMAGE_WIDTH*IMAGE_HEIGHT;
	int num_outputPixels = 26 * 26;

	fstream fs;
	string data;
	myDatatype data_myDatatype;
	myDatatype imgArray[num_pixels], groundTruthArray[num_outputPixels], outputArray[num_outputPixels];

	// Read testing image into imgArray
	fs.open("C:/Users/user/Desktop/Embbed_Application/MNIST_Inference_Device/hlsTest_float_folder/input.txt");
	for (int i = 0; i < num_pixels; i++){
		getline(fs, data);
		data_myDatatype = stof(data);
		imgArray[i] = data_myDatatype;
	}
	fs.close();

//	// Print the testing img
//	for (int y = 0; y < 28; y++){
//		for (int x = 0; x < 28; x++){
//			cout << imgArray[28*y+x];
//			if (x != 27) cout << "\t";
//			else cout << endl;
//		}
//	}

	MNIST(imgArray, outputArray);

	// Read the ground truth of conv1's output
	fs.open("C:/Users/user/Desktop/Embbed_Application/MNIST_Inference_Device/hlsTest_float_folder/conv1_output.txt");
	for (int i = 0; i < num_outputPixels; i++){
		getline(fs, data);
		data_myDatatype = stof(data);
		groundTruthArray[i] = data_myDatatype;
	}
	fs.close();


	if (pass == 0) {
		cout << "test failed" << endl;
		return 1;
	}
	cout << "test passed" << endl;
	return 0;
}
