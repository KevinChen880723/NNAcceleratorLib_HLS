#include <iostream>
#include <fstream>
#include <string>
#include "./MNIST.h"

using namespace std;

int main(int argc, char *argv[]){
	bool pass = 1;

	fstream fs;
	string data;
	myDatatype data_myDatatype;
	fs.open("C:/Users/user/Desktop/Embbed_Application/MNIST_Inference_Device/hlsTest_float_folder/input.txt");
	for (int i = 0; i < 784; i++){
		getline(fs, data);
		data_myDatatype = stof(data);
		cout << data_myDatatype << endl;
	}

	if (pass == 0) {
		cout << "test failed" << endl;
		return 1;
	}
	cout << "test passed" << endl;
	return 0;
}
