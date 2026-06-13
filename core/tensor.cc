#include "core/tensor.h"
#include "core/compute_node.h"
#include "core/ops.h"
#include "core/memory_pool.h"
#include <algorithm>
#include <cassert>
#include <unordered_map>
#include <stdexcept>
#include <numeric>

namespace autograd {

// Helper to validate the shape of a tensor
static std::vector<int> ValidateShape(std::vector<int> shape, size_t data_size)
{
  size_t expected = 1;
  for (int dim : shape) {
    if (dim <= 0) {
      throw std::invalid_argument("Tensor shape dimensions must be > 0");
    }
    expected *= static_cast<size_t>(dim);
  }
  if (expected != data_size) {
    throw std::invalid_argument(
      "Tensor data size does not match shape"
    );
  }
  return shape;
}


// --- Constructors ---

// Default constructor
Tensor::Tensor() 
: node_(std::allocate_shared<ComputeNode>(PoolAllocator<ComputeNode>(), std::vector<float>{}, std::vector<int>{}, false)) {}

// Main constructor: Moves data and shape directly into the node
Tensor::Tensor(std::vector<float> data, std::vector<int> shape, bool requires_grad) {
  ValidateShape(shape, data.size());
  // OPTIMIZATION: Use PoolAllocator and std::move
  node_ = std::allocate_shared<ComputeNode>(PoolAllocator<ComputeNode>(), std::move(data), std::move(shape), requires_grad);
}

// Private constructor: Just wraps the node
Tensor::Tensor(std::shared_ptr<ComputeNode> node) : node_(std::move(node)) {}

// --- Accessors ---
const std::vector<float>& Tensor::data() const { return node_->data; }
const std::vector<float>& Tensor::grad() const { return node_->grad; }
const std::vector<int>& Tensor::shape() const { return node_->shape; }
int Tensor::size() const {
  if (node_->shape.empty()) return 0;
  return std::accumulate(node_->shape.begin(), node_->shape.end(), 1, std::multiplies<int>());
}
std::vector<float>& Tensor::mutable_grad() { 
  node_->EnsureGrad();
  return node_->grad;
}
std::vector<float>& Tensor::mutable_data() { return node_->data; }
bool Tensor::requires_grad() const { return node_->requires_grad; }

void Tensor::zero_grad() {
  std::fill(node_->grad.begin(), node_->grad.end(), 0.0f);
}

// --- The core graph logic ---
Tensor Tensor::ApplyBinaryOp(const Tensor& a, const Tensor& b,
                             const std::string& op_name,
                             BinaryMathOp forward_op,
                             BinaryGradOp backward_op) {
  // 1. Compute Data and Shape
  std::vector<float> out_data = forward_op(a.data(), b.data(), a.shape(), b.shape());
  std::vector<int> out_shape = ComputeBroadcastShape(a.shape(), b.shape());

  bool req_grad = a.requires_grad() || b.requires_grad();

  // 2. Allocate Node (Using Pool + Move Semantics)
  auto out_node = std::allocate_shared<ComputeNode>(
    PoolAllocator<ComputeNode>(),
    std::move(out_data), 
    std::move(out_shape), 
    req_grad
  );
  out_node->op_name = op_name;

  if (req_grad) {
    out_node->parents.reserve(2); // Prevent re-alloc
    out_node->parents.push_back(a.node_);
    out_node->parents.push_back(b.node_);

    auto node_a = a.node_;
    auto node_b = b.node_;
    
    std::weak_ptr<ComputeNode> weak_out = out_node;

    // Optimization: Lambda Capture Size
    // We NO LONGER capture 'out_shape', 'shape_a', or 'shape_b' by value.
    // They are accessed via the node pointers.
    out_node->backward_fn = [weak_out, node_a, node_b, backward_op]() {
      auto out_ptr = weak_out.lock(); 
      if (!out_ptr) return;

      // Reconstruct Tensors wrappers from raw nodes (cheap, no copies)
      Tensor t_out(out_ptr);
      Tensor t_a(node_a);
      Tensor t_b(node_b);
      
      backward_op(t_out, t_a, t_b);
    };
  }
  
  return Tensor(out_node);
}

Tensor Tensor::ApplyUnaryOp(const Tensor& a,
                            const std::string& op_name,
                            UnaryMathOp forward_op,
                            UnaryGradOp backward_op) {
  std::vector<float> out_data = forward_op(a.data());
  std::vector<int> out_shape = a.shape();

  bool req_grad = a.requires_grad();
  
  auto out_node = std::allocate_shared<ComputeNode>(
      PoolAllocator<ComputeNode>(),
      std::move(out_data), 
      std::move(out_shape), 
      req_grad
  );
  out_node->op_name = op_name;

  if (req_grad) {
    out_node->parents.push_back(a.node_);
    auto node_a = a.node_;
    std::weak_ptr<ComputeNode> weak_out = out_node;
    
    // OPTIMIZATION: Small capture
    out_node->backward_fn = [weak_out, node_a, backward_op]() {
      auto out_ptr = weak_out.lock(); 
      if (!out_ptr) return;

      Tensor t_out(out_ptr);
      Tensor t_a(node_a);
      backward_op(t_out, t_a);
    };
  }
  return Tensor(out_node);
}

// --- Backward Pass (Unchanged Logic, just context update) ---
static void BuildTopo(ComputeNode* node, std::vector<ComputeNode*>& topo, std::unordered_map<ComputeNode*, Mark>& marks) {
  auto it = marks.find(node);
  if (it != marks.end()) {
    if (it->second == Mark::Temp) throw std::runtime_error("Cycle detected");
    if (it->second == Mark::Perm) return;
  }
  marks[node] = Mark::Temp;
  for (auto& parent_ptr : node->parents) {
    BuildTopo(parent_ptr.get(), topo, marks);
  }
  marks[node] = Mark::Perm;
  topo.push_back(node);
}

void Tensor::backward() {
  if (!node_->requires_grad) return;
  node_->EnsureGrad();
  std::fill(node_->grad.begin(), node_->grad.end(), 1.0f);

  topo_cache_.clear();
  marks_cache_.clear();
  BuildTopo(node_.get(), topo_cache_, marks_cache_);

  for (auto it = topo_cache_.rbegin(); it != topo_cache_.rend(); ++it) {
    if ((*it)->backward_fn) (*it)->backward_fn();
  }
}

}  // namespace autograd