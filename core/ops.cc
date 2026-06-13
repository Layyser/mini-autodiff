#include "core/tensor.h"
#include "core/compute_node.h"
#include "core/memory_pool.h"
#include "core/ops.h"
#include <algorithm> // min, max
#include <cassert>
#include <cmath> // log
#include <stdexcept>
#include <numeric>

namespace autograd {

// --- Broadcast helpers ---
// 1. Helper: Compute the result shape (e.g., [32, 10] + [1, 10] -> [32, 10])
std::vector<int> ComputeBroadcastShape(const std::vector<int>& shape_a, 
                                       const std::vector<int>& shape_b) {
  std::vector<int> out_shape;
  auto it_a = shape_a.rbegin(); 
  auto it_b = shape_b.rbegin();
  
  while (it_a != shape_a.rend() || it_b != shape_b.rend()) {
    int dim_a = (it_a != shape_a.rend()) ? *it_a : 1; // Treat missing dims as 1
    int dim_b = (it_b != shape_b.rend()) ? *it_b : 1;
    
    if (dim_a == dim_b) {
      out_shape.insert(out_shape.begin(), dim_a);
    } else if (dim_a == 1) {
      out_shape.insert(out_shape.begin(), dim_b);
    } else if (dim_b == 1) {
      out_shape.insert(out_shape.begin(), dim_a);
    } else {
      throw std::runtime_error("Incompatible dimensions for broadcasting");
    }
    
    if (it_a != shape_a.rend()) ++it_a;
    if (it_b != shape_b.rend()) ++it_b;
  }
  return out_shape;
}

namespace {

// TODO: Optimize
// 2. Helper: Map output index to input index Note: this implementation is not efficient, but done with educational purposes
size_t MapIndex(size_t flat_index, const std::vector<int>& out_shape, const std::vector<int>& in_shape) {
  size_t in_index = 0;
  size_t in_stride = 1;
  
  // Align dimensions to the right
  int dim_offset = out_shape.size() - in_shape.size();

  for (int d = out_shape.size() - 1; d >= 0; --d) {
    int out_dim_size = out_shape[d];
    int coord = flat_index % out_dim_size; // Coordinate in this dimension
    flat_index /= out_dim_size; 

    int in_d = d - dim_offset;
    if (in_d >= 0) {
      int in_dim_size = in_shape[in_d];
      // If broadcast (size 1), coord is 0. Else, matches output.
      int in_coord = (in_dim_size == 1) ? 0 : coord;
      in_index += in_coord * in_stride;
      in_stride *= in_dim_size;
    }
  }
  return in_index;
}

// Sums a large gradient down to a smaller shape (for broadcasting backward pass)
std::vector<float> SumToShape(const std::vector<float>& grad, 
                              const std::vector<int>& large_shape, 
                              const std::vector<int>& small_shape) {
  // 1. Calculate size of the small tensor
  size_t small_size = std::accumulate(small_shape.begin(), small_shape.end(), 1, std::multiplies<int>());
  
  // 2. Prepare output gradient (init to 0.0)
  std::vector<float> reduced_grad(small_size, 0.0f);
  
  // 3. Iterate over the LARGE gradient
  for (size_t i = 0; i < grad.size(); ++i) {
    // Determine which index in the small tensor this gradient element belongs to
    size_t idx = MapIndex(i, large_shape, small_shape);
    
    // Accumulate!
    reduced_grad[idx] += grad[i];
  }
  
  return reduced_grad;
}

// --- End of broadcast helpers ---

// --- Fundamental operations ---
// --- ADDITION ---
std::vector<float> AddForward(const std::vector<float>& a, 
                              const std::vector<float>& b, 
                              const std::vector<int>& shape_a,
                              const std::vector<int>& shape_b) {
  std::vector<int> out_shape = ComputeBroadcastShape(shape_a, shape_b);
  size_t total_size = std::accumulate(out_shape.begin(), out_shape.end(), 1, std::multiplies<int>());

  std::vector<float> out(total_size);
  for (size_t i = 0; i < total_size; ++i) {
    size_t idx_a = MapIndex(i, out_shape, shape_a);
    size_t idx_b = MapIndex(i, out_shape, shape_b);
    out[i] = a[idx_a] + b[idx_b];
  }
  return out;
}

void AddBackward(Tensor& out, const Tensor& a, const Tensor& b) {
  const std::vector<float>& grad_out = out.grad();
  std::vector<int> shape_out = out.shape();

  if (a.requires_grad()) {
    std::vector<float> grad_a_update;
    if (a.shape() == shape_out) grad_a_update = grad_out;
    else grad_a_update = SumToShape(grad_out, shape_out, a.shape());

    auto& ga = const_cast<Tensor&>(a).mutable_grad();
    for (size_t i = 0; i < ga.size(); ++i) ga[i] += grad_a_update[i];
  }

  if (b.requires_grad()) {
    std::vector<float> grad_b_update;
    if (b.shape() == shape_out) grad_b_update = grad_out;
    else grad_b_update = SumToShape(grad_out, shape_out, b.shape());

    auto& gb = const_cast<Tensor&>(b).mutable_grad();
    for (size_t i = 0; i < gb.size(); ++i) gb[i] += grad_b_update[i];
  }
}

// --- SUBSTRACTION ---
std::vector<float> SubForward(const std::vector<float>& a, 
                              const std::vector<float>& b,
                              const std::vector<int>& shape_a,
                              const std::vector<int>& shape_b) {
  std::vector<int> out_shape = ComputeBroadcastShape(shape_a, shape_b);
  size_t total_size = std::accumulate(out_shape.begin(), out_shape.end(), 1, std::multiplies<int>());

  std::vector<float> out(total_size);
  for (size_t i = 0; i < total_size; ++i) {
    size_t idx_a = MapIndex(i, out_shape, shape_a);
    size_t idx_b = MapIndex(i, out_shape, shape_b);
    out[i] = a[idx_a] - b[idx_b];
  }
  return out;
}

void SubBackward(Tensor& out, const Tensor& a, const Tensor& b) {
  const std::vector<float>& grad_out = out.grad();
  std::vector<int> shape_out = out.shape();

  if (a.requires_grad()) {
    std::vector<float> grad_a_update;
    if (a.shape() == shape_out) grad_a_update = grad_out;
    else grad_a_update = SumToShape(grad_out, shape_out, a.shape());

    auto& ga = const_cast<Tensor&>(a).mutable_grad();
    for (size_t i = 0; i < ga.size(); ++i) ga[i] += grad_a_update[i];
  }

  if (b.requires_grad()) {
    std::vector<float> grad_b_update;
    if (b.shape() == shape_out) grad_b_update = grad_out;
    else grad_b_update = SumToShape(grad_out, shape_out, b.shape());

    auto& gb = const_cast<Tensor&>(b).mutable_grad();
    for (size_t i = 0; i < gb.size(); ++i) gb[i] -= grad_b_update[i];
  }
}

// --- MULTIPLICATION ---
std::vector<float> MulForward(const std::vector<float>& a, 
                              const std::vector<float>& b,
                              const std::vector<int>& shape_a,
                              const std::vector<int>& shape_b) {
  std::vector<int> out_shape = ComputeBroadcastShape(shape_a, shape_b);
  size_t total_size = std::accumulate(out_shape.begin(), out_shape.end(), 1, std::multiplies<int>());

  std::vector<float> out(total_size);
  for (size_t i = 0; i < total_size; ++i) {
    size_t idx_a = MapIndex(i, out_shape, shape_a);
    size_t idx_b = MapIndex(i, out_shape, shape_b);
    out[i] = a[idx_a] * b[idx_b];
  }
  return out;
}

void MulBackward(Tensor& out, const Tensor& a, const Tensor& b) {
  const auto& grad_out = out.grad();
  std::vector<int> shape_out = out.shape();
  if (a.requires_grad()) {
    // 1. Compute the expanded gradient (dL/dY * B)
    std::vector<float> temp_grad(grad_out.size());
    const auto& b_data = b.data();
    for (size_t i = 0; i < grad_out.size(); ++i) {
      size_t idx_b = MapIndex(i, shape_out, b.shape()); // Virtually expand B
      temp_grad[i] = grad_out[i] * b_data[idx_b];
    }

    // 2. Reduce (Sum) if A is smaller than output
    std::vector<float> final_grad_a;
    if (a.shape() == shape_out) final_grad_a = temp_grad;
    else final_grad_a = SumToShape(temp_grad, shape_out, a.shape());

    // 3. Sum gradients to A
    auto& ga = const_cast<Tensor&>(a).mutable_grad();
    for (size_t i = 0; i < ga.size(); ++i) ga[i] += final_grad_a[i];
  }

  if (b.requires_grad()) {
    // Symmetric explanation
    std::vector<float> temp_grad(grad_out.size());
    const auto& a_data = a.data();
    for (size_t i = 0; i < grad_out.size(); ++i) {
      size_t idx_a = MapIndex(i, shape_out, a.shape());
      temp_grad[i] = grad_out[i] * a_data[idx_a];
    }

    std::vector<float> final_grad_b;
    if (b.shape() == shape_out) final_grad_b = temp_grad;
    else final_grad_b = SumToShape(temp_grad, shape_out, b.shape());

    auto& gb = const_cast<Tensor&>(b).mutable_grad();
    for (size_t i = 0; i < gb.size(); ++i) gb[i] += final_grad_b[i];
  }
}

// --- DIVISION ---

// Throws if any element of the divisor is zero, instead of silently
// producing inf/NaN that then propagates through forward and backward.
void CheckNoZeroDivisor(const std::vector<float>& b) {
  for (float val : b) {
    if (val == 0.0f) {
      throw std::invalid_argument("division by zero: divisor tensor contains a zero element");
    }
  }
}

std::vector<float> DivForward(const std::vector<float>& a,
                              const std::vector<float>& b,
                              const std::vector<int>& shape_a,
                              const std::vector<int>& shape_b) {
  CheckNoZeroDivisor(b);

  std::vector<int> out_shape = ComputeBroadcastShape(shape_a, shape_b);
  size_t total_size = std::accumulate(out_shape.begin(), out_shape.end(), 1, std::multiplies<int>());

  std::vector<float> out(total_size);
  for (size_t i = 0; i < total_size; ++i) {
    size_t idx_a = MapIndex(i, out_shape, shape_a);
    size_t idx_b = MapIndex(i, out_shape, shape_b);
    out[i] = a[idx_a] / b[idx_b];
  }
  return out;
}

void DivBackward(Tensor& out, const Tensor& a, const Tensor& b) {
  const auto& grad_out = out.grad();
  std::vector<int> shape_out = out.shape();
  if (a.requires_grad()) {
    std::vector<float> temp_grad(grad_out.size());
    const auto& b_data = b.data();
    for (size_t i = 0; i < grad_out.size(); ++i) {
      size_t idx_b = MapIndex(i, shape_out, b.shape());
      temp_grad[i] = grad_out[i] / b_data[idx_b];
    }

    std::vector<float> final_grad_a;
    if (a.shape() == shape_out) final_grad_a = temp_grad;
    else final_grad_a = SumToShape(temp_grad, shape_out, a.shape());

    auto& ga = const_cast<Tensor&>(a).mutable_grad();
    for (size_t i = 0; i < ga.size(); ++i) ga[i] += final_grad_a[i];
  }

  if (b.requires_grad()) {
    std::vector<float> temp_grad(grad_out.size());
    const auto& a_data = a.data();
    const auto& b_data = b.data();
    for (size_t i = 0; i < grad_out.size(); ++i) {
      size_t idx_a = MapIndex(i, shape_out, a.shape());
      size_t idx_b = MapIndex(i, shape_out, b.shape());
      temp_grad[i] = grad_out[i] * a_data[idx_a] / (b_data[idx_b] * b_data[idx_b]);
    }

    std::vector<float> final_grad_b;
    if (b.shape() == shape_out) final_grad_b = temp_grad;
    else final_grad_b = SumToShape(temp_grad, shape_out, b.shape());

    auto& gb = const_cast<Tensor&>(b).mutable_grad();
    for (size_t i = 0; i < gb.size(); ++i) gb[i] -= final_grad_b[i];
  }
}

// --- MATMUL ---
// A: shape [M, K]
// B: shape [K, N]
// Out: shape [M, N]

// Validates that both shapes are 2D and that inner dimensions match
void ValidateMatMulShapes(const std::vector<int>& shape_a, const std::vector<int>& shape_b) {
  if (shape_a.size() != 2 || shape_b.size() != 2) {
    throw std::invalid_argument("matmul: both tensors must be 2D");
  }
  if (shape_a[1] != shape_b[0]) {
    throw std::invalid_argument("matmul: inner dimensions must match (A is [M,K], B must be [K,N])");
  }
}

std::vector<float> MatMulForward(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 const std::vector<int>& shape_a,
                                 const std::vector<int>& shape_b) {
  // Assume a is (M, K) and b is (K, N)
  int M = shape_a[0];
  int K = shape_a[1];
  int N = shape_b[1];
  
  std::vector<float> out(M * N, 0.0f);
  
  // Reordered for cache performance (ijk -> ikj)
  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      float a_ik = a[i * K + k];
      for (int j = 0; j < N; ++j) {
        out[i * N + j] += a_ik * b[k * N + j];
      }
    }
  }
  
  return out;
}

void MatMulBackward(Tensor& out, const Tensor& a, const Tensor& b) {
  const auto& grad_out = out.grad();  // shape: (M, N)
  const auto& shape_a = a.shape();
  const auto& shape_b = b.shape();
  
  int M = shape_a[0];
  int K = shape_a[1];
  int N = shape_b[1];
  
  if (a.requires_grad()) {
    auto& ga = const_cast<Tensor&>(a).mutable_grad();
    const auto& b_data = b.data();
    
    for (int i = 0; i < M; ++i) {
      for (int k = 0; k < K; ++k) {
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
          sum += grad_out[i * N + j] * b_data[k * N + j];
        }
        ga[i * K + k] += sum;
      }
    }
  }
  
  if (b.requires_grad()) {
    auto& gb = const_cast<Tensor&>(b).mutable_grad();
    const auto& a_data = a.data();
    
    for (int k = 0; k < K; ++k) {
      for (int j = 0; j < N; ++j) {
        float sum = 0.0f;
        for (int i = 0; i < M; ++i) {
          sum += a_data[i * K + k] * grad_out[i * N + j];
        }
        gb[k * N + j] += sum;
      }
    }
  }
}

// --- NEGATION ---
std::vector<float> NegForward(const std::vector<float>& a) {
  std::vector<float> out(a.size());
  for (size_t i = 0; i < a.size(); ++i) out[i] = -a[i];
  return out;
}

void NegBackward(Tensor& out, const Tensor& a) {
  const auto& grad_out = out.grad();
  if (a.requires_grad()) {
    auto& ga = const_cast<Tensor&>(a).mutable_grad();
    for (size_t i = 0; i < ga.size(); ++i) ga[i] -= grad_out[i];
  }
}

// --- RELU ---
std::vector<float> ReluForward(const std::vector<float>& a) {
  std::vector<float> out(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    out[i] = (a[i] > 0.0f) ? a[i] : 0.0f;
  }
  return out;
}

void ReluBackward(Tensor& out, const Tensor& a) {
  const auto& grad_out = out.grad();
  if (a.requires_grad()) {
    auto& ga = const_cast<Tensor&>(a).mutable_grad();
    const auto& input_data = a.data();
    for (size_t i = 0; i < ga.size(); ++i) {
      float derivative = (input_data[i] > 0.0f) ? 1.0f : 0.0f;
      ga[i] += grad_out[i] * derivative;
    }
  }
}

// --- SIGMOID ---
std::vector<float> SigmoidForward(const std::vector<float>& a) {
  std::vector<float> out(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    out[i] = 1.0f / (1.0f + std::exp(-a[i]));
  }
  return out;
}

void SigmoidBackward(Tensor& out, const Tensor& a) {
  const auto& grad_out = out.grad();
  if (a.requires_grad()) {
    auto& ga = const_cast<Tensor&>(a).mutable_grad();
    const auto& y = out.data(); // Optimization: Derivative uses output (y), not input (a)
    for (size_t i = 0; i < ga.size(); ++i) {
       float derivative = y[i] * (1.0f - y[i]);
       ga[i] += grad_out[i] * derivative;
    }
  }
}

// --- NATURAL LOG ---
std::vector<float> LogForward(const std::vector<float>& a) {
  std::vector<float> out(a.size());
  for (size_t i = 0; i < a.size(); ++i) out[i] = std::log(a[i]);
  return out;
}

void LogBackward(Tensor& out, const Tensor& a) {
  const auto& grad_out = out.grad();
  if (a.requires_grad()) {
    auto& ga = const_cast<Tensor&>(a).mutable_grad();
    const auto& da = a.data();
    for (size_t i = 0; i < ga.size(); ++i) ga[i] += grad_out[i] / da[i];
  }
}

// --- CLAMP ---
std::vector<float> ClampForward(const std::vector<float>& a, float min_val, float max_val) {
  std::vector<float> out(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    out[i] = std::min(std::max(a[i], min_val), max_val);
  }
  return out;
}

void ClampBackward(Tensor& out, const Tensor& a, float min_val, float max_val) {
  const auto& grad_out = out.grad();
  if (a.requires_grad()) {
    auto& ga = const_cast<Tensor&>(a).mutable_grad();
    const auto& da = a.data();
    for (size_t i = 0; i < ga.size(); ++i) {
      // Gradient is 1 inside the clamp range, 0 outside (flat regions)
      float derivative = (da[i] >= min_val && da[i] <= max_val) ? 1.0f : 0.0f;
      ga[i] += grad_out[i] * derivative;
    }
  }
}

// --- SUM --- Authored by Sonnet-4.5
std::vector<float> SumForward(const std::vector<float>& a) {
  float total = 0.0f;
  for (float val : a) total += val;
  return {total};  // Single element vector
}

void SumBackward(Tensor& out, const Tensor& a) {
  // Gradient of sum: distributes gradient equally to all inputs
  // d/dx (sum(x)) = 1 for each element
  const auto& grad_out = out.grad();
  
  if (a.requires_grad()) {
    auto& ga = const_cast<Tensor&>(a).mutable_grad();
    // Broadcast the single gradient value to all input elements
    for (size_t i = 0; i < ga.size(); ++i) {
      ga[i] += grad_out[0];  // grad_out is scalar, so index [0]
    }
  }
}

// --- MEAN --- Authored by Sonnet-4.5
std::vector<float> MeanForward(const std::vector<float>& a) {
  float total = 0.0f;
  for (float val : a) total += val;
  return {total / a.size()};
}

void MeanBackward(Tensor& out, const Tensor& a) {
  const auto& grad_out = out.grad();
  if (a.requires_grad()) {
    auto& ga = const_cast<Tensor&>(a).mutable_grad();
    float grad_value = grad_out[0] / a.data().size();  // Divide by N
    
    for (size_t i = 0; i < ga.size(); ++i) {
      ga[i] += grad_value;
    }
  }
}

// --- End of fundamental operations ---


} // namespace


// == BINARY OPERATORS ==
Tensor operator+(const Tensor& a, const Tensor& b) {
  return Tensor::ApplyBinaryOp(a, b, "Add", AddForward, AddBackward);
}

Tensor operator-(const Tensor& a, const Tensor& b) {
  return Tensor::ApplyBinaryOp(a, b, "Sub", SubForward, SubBackward);
}

Tensor operator*(const Tensor& a, const Tensor& b) {
  return Tensor::ApplyBinaryOp(a, b, "Mul", MulForward, MulBackward);
}

Tensor operator/(const Tensor& a, const Tensor& b) {
  return Tensor::ApplyBinaryOp(a, b, "Div", DivForward, DivBackward);
}

Tensor matmul(const Tensor& a, const Tensor& b) { // Specific case where we need shapes of the Tensors
  ValidateMatMulShapes(a.shape(), b.shape());

  // 1. Compute Data and Shape
  std::vector<float> out_data = MatMulForward(a.data(), b.data(), a.shape(), b.shape());
  std::vector<int> out_shape = {a.shape()[0], b.shape()[1]};

  bool req_grad = a.requires_grad() || b.requires_grad();

  // 2. Allocate Node (Using Pool + Move Semantics)
  auto out_node = std::allocate_shared<ComputeNode>(
    PoolAllocator<ComputeNode>(),
    std::move(out_data), 
    std::move(out_shape), 
    req_grad
  );
  out_node->op_name = "matmul";

  if (req_grad) {
    out_node->parents.reserve(2); // Prevent re-alloc
    out_node->parents.push_back(a.node_);
    out_node->parents.push_back(b.node_);

    auto node_a = a.node_;
    auto node_b = b.node_;
    
    std::weak_ptr<ComputeNode> weak_out = out_node;

    // OPTIMIZATION: Lambda Capture Size
    // We NO LONGER capture 'out_shape', 'shape_a', or 'shape_b' by value.
    // They are accessed via the node pointers.
    auto backward_op = MatMulBackward;
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

// == UNARY OPERATORS ==
Tensor operator-(const Tensor& a) {
  return Tensor::ApplyUnaryOp(a, "Neg", NegForward, NegBackward);
}

Tensor relu(const Tensor& a) {
  return Tensor::ApplyUnaryOp(a, "Relu", ReluForward, ReluBackward);
}

Tensor sigmoid(const Tensor& a) {
  return Tensor::ApplyUnaryOp(a, "Sigmoid", SigmoidForward, SigmoidBackward);
}

Tensor log(const Tensor& a) {
  return Tensor::ApplyUnaryOp(a, "Log", LogForward, LogBackward);
}

Tensor clamp(const Tensor& a, float min_val, float max_val) {
  auto forward_op = [min_val, max_val](const std::vector<float>& x) {
    return ClampForward(x, min_val, max_val);
  };
  auto backward_op = [min_val, max_val](Tensor& out, const Tensor& x) {
    ClampBackward(out, x, min_val, max_val);
  };
  return Tensor::ApplyUnaryOp(a, "Clamp", forward_op, backward_op);
}

Tensor sum(const Tensor& a) {
  return Tensor::ApplyUnaryOp(a, "Sum", SumForward, SumBackward);
}

Tensor mean(const Tensor& a) {
  return Tensor::ApplyUnaryOp(a, "Mean", MeanForward, MeanBackward);
}

}  // namespace autograd