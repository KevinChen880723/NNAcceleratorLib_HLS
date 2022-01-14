#include <iostream>
#include "MNIST.h"

using namespace std;

int main(int argc, char *argv[]){
	int a, b, c;
	int i, j;
	bool pass = 1;
	for(a = 0; a < 10; a++){
		for(b = 0; b < 10; b++){
			MNIST(a, b, &c);
			cout << a << " + " << b << " = " << c << endl;
			if (c != a + b) pass = 0;
		}
	}
	if (pass == 0) {
		cout << "test failed" << endl;
		return 1;
	}
	cout << "test passed" << endl;
	return 0;
}
