#include <iostream>
#include <assert.h>

#include "include/ull128.h"


using namespace std;
void test_lshift()
{
	ulonglong2 val = make_ulonglong2(1, 0);
	unsigned long long int real = 1;
	for (int i = 0; i < 64; i++){
		val = lshiftull2(val, 1);
		real <<= 1;
		if (val.x != real){
			cout << "lshiftull2 FAILED: " << i << " " << val.x << " " << val.y << " " << real << " exiting..." << endl;
			exit(-1);
		}
	}

	real = 1;
	for (int i = 0; i < 64; i++){
		val = lshiftull2(val, 1);
		real <<= 1;
		if (val.y != real){
			cout << "lshiftull2 FAILED: " << i << " " << val.x << " " << val.y << " " << real << " exiting..." << endl;
			exit(-1);
		}

	}
	cout << "TEST lshiftull2 2 PASS" << endl;

	val.x = -1;
	real = -1;
	for (int i = 0; i < 64; i++){
		val = lshiftull2(val, 1);
		real <<= 1;
		if (val.x != real){
			cout << "lshiftull2 FAILED: " << i << " " << val.x << " " << val.y << " " << real << " exiting..." << endl;
			exit(-1);

		}
	}

	real = -1;
	for (int i = 0; i < 64; i++){
		val = lshiftull2(val, 1);
		real <<= 1;
		if (val.y != real){
			cout << "lshiftull2 FAILED: " << i << " " << val.x << " " << val.y << " " << real << " exiting..." << endl;
			exit(-1);

		}
	}

	cout << "TEST lshiftull2 2 PASS" << endl;

	for (int i = 0; i < 32; i++){
		ulonglong2 ret = lshiftull2(val, i);
		unsigned long long int ret2 = real << i;
		if (val.x != real){
			cout << "lshiftull2 FAILED: " << i << " " << val.x << " " << val.y << " " << real << " exiting..." << endl;

			exit(-1);
		}
	}
	cout << "lshiftull2 SUCCESS!" << endl;
}

void test_rshift()
{

	ulonglong2 val = make_ulonglong2(0,0x1000000000000000);
	unsigned long long int real = 0x1000000000000000;
	for (int i = 0; i < 64; i++){
		val = rshiftull2(val, 1);
		real >>= 1;
		if (val.y != real){
			cout << "rshiftull2 FAILED: " << i << " " << val.x << " " << val.y << " " << real << " exiting..." << endl;
			exit(-1);
		}
	}

	real = 0x1000000000000000;
	for (int i = 0; i < 64; i++){
		val = rshiftull2(val, 1);
		real >>= 1;
		if (val.x != real){
			cout << "rshiftull2 FAILED: " << i << " " << val.x << " " << val.y << " " << real << " exiting..." << endl;
			exit(-1);
		}

	}
	cout << "TEST rshiftull2 1 PASS" << endl;

	val.y = -1;
	real = -1;
	for (int i = 0; i < 64; i++){
		val = rshiftull2(val, 1);
		real >>= 1;
		if (val.y != real){
			cout << "rshiftull2 FAILED: " << i << " " << val.x << " " << val.y << " " << real << " exiting..." << endl;
			exit(-1);

		}
	}

	real = -1;
	for (int i = 0; i < 64; i++){
		val = rshiftull2(val, 1);
		real >>= 1;
		if (val.x != real){
			cout << "rshiftull2 FAILED: " << i << " " << val.x << " " << val.y << " " << real << " exiting..." << endl;
			exit(-1);

		}
	}
	cout << "TEST rshiftull2 2 PASS" << endl;


	for (int i = 0; i < 32; i++){
		ulonglong2 ret = rshiftull2(val, i);
		unsigned long long int ret2 = real >> i;
		if (val.x != real){
			cout << "rshiftull2 FAILED: " << i << " " << val.x << " " << val.y << " " << real << " exiting..." << endl;

			exit(-1);
		}
	}
	cout << "rshiftull2 SUCCESS!" << endl;
}
int main()
{
	
	test_lshift();
	test_rshift();


}