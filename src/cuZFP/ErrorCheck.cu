#include "ErrorCheck.h"
#include <iostream>
#include <cuda.h>
using namespace std;

ErrorCheck::ErrorCheck()
{

}

void ErrorCheck::chk(std::string msg)
{
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        cout << msg << " : " << error;
        cout << " " << cudaGetErrorString(error) << endl;
    }
}

void ErrorCheck::chk()
{
    chk(str.str());
    str.str("");
}
