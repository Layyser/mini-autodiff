#include "tensor.h"
#include <algorithm>
#include "compute_node.h"
#include <cassert>
#include <unordered_map>
#include <stdexcept>
#include <numeric> // accumulate

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
Tensor::Tensor() : node_(std::make_shared<ComputeNode>(std::vector<float>{}, false)) {}

Tensor::Tensor(const std::vector<float>& data, std::vector<int> shape, bool requires_grad)
  : shape_(ValidateShape(std::move(shape), data.size())), node_(std::make_shared<ComputeNode>(data, requires_grad)) {}

// Private constructor used by ApplyOp to wrap a new node
Tensor::Tensor(std::vector<int> shape, std::shared_ptr<ComputeNode> node) 
  : shape_(std::move(shape)), node_(std::move(node)) {}

// --- Accessors ---
const std::vector<float>& Tensor::data() const { return node_->data; }
const std::vector<float>& Tensor::grad() const { return node_->grad; }
const std::vector<int>& Tensor::shape() const { return shape_; }
int Tensor::size() const {
  if (shape_.empty()) return 0;
  return std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>());
}
std::vector<float>& Tensor::mutable_grad() { return node_->grad; }
bool Tensor::requires_grad() const { return node_->requires_grad; }

void Tensor::zero_grad() {
  std::fill(node_->grad.begin(), node_->grad.end(), 0.0f);
}

// --- The core graph logic ---
Tensor Tensor::ApplyBinaryOp(const Tensor& a, const Tensor& b,
                             const std::string& op_name,
                             BinaryMathOp forward_op,
                             BinaryGradOp backward_op) {
  std::vector<float> out_data = forward_op(a.data(), b.data());
  std::vector<int> out_shape = a.shape(); // Assuming same shape for now

  bool req_grad = a.requires_grad() || b.requires_grad();
  auto out_node = std::make_shared<ComputeNode>(out_data, req_grad);
  out_node->op_name = op_name;

  if (req_grad) {
    out_node->parents.push_back(a.node_);
    out_node->parents.push_back(b.node_);

    auto node_a = a.node_;
    auto node_b = b.node_;
    
    auto shape_a = a.shape_; 
    auto shape_b = b.shape_;
    
    // Use weak_ptr to break the cycle
    std::weak_ptr<ComputeNode> weak_out = out_node;

    out_node->backward_fn = [weak_out, node_a, node_b, backward_op, out_shape, shape_a, shape_b]() {
      auto out_ptr = weak_out.lock(); 
      if (!out_ptr) return;

      Tensor t_out(out_shape, out_ptr);
      Tensor t_a(shape_a, node_a);
      Tensor t_b(shape_b, node_b);
      backward_op(t_out, t_a, t_b);
    };
  }
  return Tensor(out_shape, out_node);
}

Tensor Tensor::ApplyUnaryOp(const Tensor& a,
                            const std::string& op_name,
                            UnaryMathOp forward_op,
                            UnaryGradOp backward_op) {
  std::vector<float> out_data = forward_op(a.data());
  std::vector<int> out_shape = a.shape(); // Assuming same shape for now

  bool req_grad = a.requires_grad();
  auto out_node = std::make_shared<ComputeNode>(out_data, req_grad);
  out_node->op_name = op_name;

  if (req_grad) {
    out_node->parents.push_back(a.node_);
    
    auto node_a = a.node_;
    auto shape_a = a.shape_; 

    std::weak_ptr<ComputeNode> weak_out = out_node;
    
    out_node->backward_fn = [weak_out, node_a, backward_op, out_shape, shape_a]() {
      auto out_ptr = weak_out.lock(); 
      if (!out_ptr) return; // Node already destroyed

      Tensor t_out(out_shape, out_ptr);
      Tensor t_a(shape_a, node_a);
      backward_op(t_out, t_a);
    };
  }
  return Tensor(out_shape, out_node);
}

// --- Backward Pass ---

// Topological sort with cycle detection
enum class Mark { None, Temp, Perm };

static void BuildTopo(
  ComputeNode* node,
  std::vector<ComputeNode*>& topo,
  std::unordered_map<ComputeNode*, Mark>& marks
) {
  auto it = marks.find(node);

  if (it != marks.end()) {
    if (it->second == Mark::Temp) {
      throw std::runtime_error("Cycle detected in autograd graph");
    }
    if (it->second == Mark::Perm) {
      return; // already processed
    }
  }

  marks[node] = Mark::Temp;

  for (auto& parent_ptr : node->parents) {
    BuildTopo(parent_ptr.get(), topo, marks);
  }

  // Mark as fully processed
  marks[node] = Mark::Perm;
  topo.push_back(node);
}

void Tensor::backward() {
  if (!node_->requires_grad) return;

  // Seed gradient (defaults to 1.0)
  std::fill(node_->grad.begin(), node_->grad.end(), 1.0f);

  std::vector<ComputeNode*> topo;
  std::unordered_map<ComputeNode*, Mark> marks;
  
  BuildTopo(node_.get(), topo, marks);

  // Traverse in reverse topological order
  for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
    if ((*it)->backward_fn) {
      (*it)->backward_fn();
    }
  }
}

}  // namespace autograd