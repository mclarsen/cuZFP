#ifndef cuZFP_h
#define cuZFP_h

#include<vector>

namespace cuZFP {

extern "C" void encode_vector(std::vector<double> &in_data, std::vector<int> &encoded_data);

} // namespace cuZFP

#endif
