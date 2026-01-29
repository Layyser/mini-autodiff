#include "tensor.h"
#include "compute_node.h"
#include <cassert>
#include <cmath> // log

namespace autograd {

namespace {

// --- ADDITION ---
std::vector<float> AddForward(const std::vector<float>& a, const std::vector<float>& b) {
  assert(a.size() == b.size());
  std::vector<float> out(a.size());
  for (size_t i = 0; i < a.size(); ++i) out[i] = a[i] + b[i];
  return out;
}

void AddBackward(Tensor& out, const Tensor& a, const Tensor& b) {
  const auto& grad_out = out.grad();
  if (a.requires_grad()) {
    auto& ga = const_cast<Tensor&>(a).mutable_grad();
    for (size_t i = 0; i < ga.size(); ++i) ga[i] += grad_out[i];
  }
  if (b.requires_grad()) {
    auto& gb = const_cast<Tensor&>(b).mutable_grad();
    for (size_t i = 0; i < gb.size(); ++i) gb[i] += grad_out[i];
  }
}

// --- SUBSTRACTION ---
std::vector<float> SubForward(const std::vector<float>& a, const std::vector<float>& b) {
  assert(a.size() == b.size());
  std::vector<float> out(a.size());
  for (size_t i = 0; i < a.size(); ++i) out[i] = a[i] - b[i];
  return out;
}

void SubBackward(Tensor& out, const Tensor& a, const Tensor& b) {
  const auto& grad_out = out.grad();
  if (a.requires_grad()) {
    auto& ga = const_cast<Tensor&>(a).mutable_grad();
    for (size_t i = 0; i < ga.size(); ++i) ga[i] += grad_out[i];
  }
  if (b.requires_grad()) {
    auto& gb = const_cast<Tensor&>(b).mutable_grad();
    for (size_t i = 0; i < gb.size(); ++i) gb[i] -= grad_out[i];
  }
}

// --- MULTIPLICATION ---
std::vector<float> MulForward(const std::vector<float>& a, const std::vector<float>& b) {
  assert(a.size() == b.size());
  std::vector<float> out(a.size());
  for (size_t i = 0; i < a.size(); ++i) out[i] = a[i] * b[i];
  return out;
}

void MulBackward(Tensor& out, const Tensor& a, const Tensor& b) {
  const auto& grad_out = out.grad();
  if (a.requires_grad()) {
    auto& ga = const_cast<Tensor&>(a).mutable_grad();
    const auto& db = b.data();
    for (size_t i = 0; i < ga.size(); ++i) ga[i] += grad_out[i] * db[i];
  }
  if (b.requires_grad()) {
    auto& gb = const_cast<Tensor&>(b).mutable_grad();
    const auto& da = a.data();
    for (size_t i = 0; i < gb.size(); ++i) gb[i] += grad_out[i] * da[i];
  }
}

// --- DIVISION ---
std::vector<float> DivForward(const std::vector<float>& a, const std::vector<float>& b) {
  assert(a.size() == b.size());
  std::vector<float> out(a.size());
  for (size_t i = 0; i < a.size(); ++i) out[i] = a[i] / b[i];
  return out;
}

void DivBackward(Tensor& out, const Tensor& a, const Tensor& b) {
  const auto& grad_out = out.grad();
  if (a.requires_grad()) {
    auto& ga = const_cast<Tensor&>(a).mutable_grad();
    const auto& db = b.data();
    for (size_t i = 0; i < ga.size(); ++i) ga[i] += grad_out[i] / db[i];
  }
  if (b.requires_grad()) {
    auto& gb = const_cast<Tensor&>(b).mutable_grad();
    const auto& da = a.data();
    const auto& db = b.data();
    for (size_t i = 0; i < gb.size(); ++i) gb[i] -= grad_out[i] * da[i] / (db[i] * db[i]);
  }
}

// --- MATMUL ---
// A: shape [M, K]
// B: shape [K, N]
// Out: shape [M, N]
std::vector<float> MatMulForward(const std::vector<float>& a, const std::vector<float>& b,
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
  std::vector<float> out_data = MatMulForward(a.data(), b.data(), a.shape(), b.shape());
  std::vector<int> out_shape = {a.shape()[0], b.shape()[1]};
  
  bool req_grad = a.requires_grad() || b.requires_grad();
  auto out_node = std::make_shared<ComputeNode>(out_data, req_grad);
  out_node->op_name = "matmul";
  
  // 4. Connect Graph
  if (req_grad) {
    out_node->parents.push_back(a.node_);
    out_node->parents.push_back(b.node_);
    
    auto node_a = a.node_;
    auto node_b = b.node_;
    
    auto shape_a = a.shape_;
    auto shape_b = b.shape_;

    // Use weak_ptr to break the cycle
    std::weak_ptr<ComputeNode> weak_out = out_node;
    
    auto backward_op = MatMulBackward;
    out_node->backward_fn = [weak_out, node_a, node_b, backward_op, out_shape, shape_a, shape_b]() {
      auto out_ptr = weak_out.lock(); 
      if (!out_ptr) return; // Node already destroyed

      Tensor t_out(out_shape, out_ptr);
      Tensor t_a(shape_a, node_a);
      Tensor t_b(shape_b, node_b);
      backward_op(t_out, t_a, t_b);
    };
  }
  
  return Tensor(out_shape, out_node);
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

}  // namespace autograd