#ifndef ERRORCHECK_H
#define ERRORCHECK_H
#include <string>
#include <sstream>

using std::stringstream;
class ErrorCheck
{
public:
    ErrorCheck();
    void chk(std::string msg);
    void chk();
    cudaError error;
    stringstream str;
};

#endif // ERRORCHECK_H
