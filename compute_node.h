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
  std::vector<int> shape; // Optimization: Shape is now stored here, it is more performant although not that intuitive
  bool requires_grad;
  
  std::string op_name;
  std::vector<std::shared_ptr<ComputeNode>> parents;
  std::function<void()> backward_fn;
  
  // Optimization: Pass by value to allow std::move
  ComputeNode(std::vector<float> data_, std::vector<int> shape_, bool requires_grad_)
    : data(std::move(data_)), shape(std::move(shape_)), requires_grad(requires_grad_) {}

  inline void EnsureGrad() {
    if (grad.empty()) {
      grad.resize(data.size(), 0.0f);
    }
  }
};

}  // namespace autograd

#endif  // AUTOGRAD_COMPUTE_NODE_H_