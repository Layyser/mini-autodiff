// ops.h
#ifndef AUTOGRAD_OPS_H_
#define AUTOGRAD_OPS_H_

#include <vector>

namespace autograd {

std::vector<int> ComputeBroadcastShape(const std::vector<int>& shape_a, const std::vector<int>& shape_b);

}  // namespace autograd

#endif  // AUTOGRAD_OPS_H_