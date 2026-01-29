#ifndef AUTOGRAD_COMPUTE_NODE_H_
#define AUTOGRAD_COMPUTE_NODE_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace autograd {

struct ComputeNode {
  std::vector<float> data;
  std::vector<float> grad;
  std::vector<int> shape;
  bool requires_grad;
  
  std::string op_name;
  std::vector<std::shared_ptr<ComputeNode>> parents;
  std::function<void()> backward_fn;
  
  ComputeNode(const std::vector<float>& data_, bool requires_grad_) 
            : data(data_), grad(data_.size(), 0.0f), requires_grad(requires_grad_) {}
};

}  // namespace autograd

#endif  // AUTOGRAD_COMPUTE_NODE_H_