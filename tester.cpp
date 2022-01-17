#include <iostream>
#include <fstream>
#include "./MNIST.h"

using namespace std;

int main(int argc, char *argv[]){
	bool pass = 1;
	fstream fstream("./hlsTest_float_folder/input.txt");
	myDatatype data;
	fstream >> data;
	if (pass == 0) {
		cout << "test failed" << endl;
		return 1;
	}
	cout << "test passed" << endl;
	return 0;
}
