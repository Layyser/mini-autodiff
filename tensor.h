#ifndef AUTOGRAD_TENSOR_H_
#define AUTOGRAD_TENSOR_H_

#include <functional>
#include <memory> // for std::shared_ptr
#include <string>
#include <vector>
#include <unordered_map>

namespace autograd {

// Marking states for topological sort
enum class Mark { None, Temp, Perm };

// Forward declaration of the implementation struct
struct ComputeNode;

class Tensor {
 public:
  Tensor();
  explicit Tensor(std::vector<float> data, std::vector<int> shape, bool requires_grad = false);

  // -- Accessors --
  const std::vector<float>& data() const;
  const std::vector<float>& grad() const;
  const std::vector<int>& shape() const;
  int size() const;
  std::vector<float>& mutable_grad();
  std::vector<float>& mutable_data();
  bool requires_grad() const;

  // -- Engine API --
  void backward();
  void zero_grad();

  // -- Operators --
  friend Tensor operator+(const Tensor& a, const Tensor& b);
  friend Tensor operator-(const Tensor& a, const Tensor& b);
  friend Tensor operator*(const Tensor& a, const Tensor& b);
  friend Tensor operator/(const Tensor& a, const Tensor& b);
  friend Tensor matmul(const Tensor& a, const Tensor& b);
  
  // -- Unary Operators --
  friend Tensor operator-(const Tensor& a);
  friend Tensor relu(const Tensor& a);
  friend Tensor sigmoid(const Tensor& a);
  friend Tensor log(const Tensor& a);
  friend Tensor sum(const Tensor& a);
  friend Tensor mean(const Tensor& a);

 private:
  // We need a private constructor that wraps an existing node
  explicit Tensor(std::shared_ptr<ComputeNode> node);

  using BinaryMathOp = std::function<std::vector<float>(const std::vector<float>&, const std::vector<float>&, const std::vector<int>&, const std::vector<int>&)>;
  using BinaryGradOp = std::function<void(Tensor& out, const Tensor& a, const Tensor& b)>;
  using UnaryMathOp = std::function<std::vector<float>(const std::vector<float>&)>;
  using UnaryGradOp = std::function<void(Tensor& out, const Tensor& a)>;

  static Tensor ApplyBinaryOp(const Tensor& a, const Tensor& b,
                              const std::string& op_name,
                              BinaryMathOp forward_op,
                              BinaryGradOp backward_op);

  static Tensor ApplyUnaryOp(const Tensor& a,
                             const std::string& op_name,
                             UnaryMathOp forward_op,
                             UnaryGradOp backward_op);


  std::shared_ptr<ComputeNode> node_;

  // Optimization: Don't allocate these on every backward call
  mutable std::vector<ComputeNode*> topo_cache_;
  mutable std::unordered_map<ComputeNode*, autograd::Mark> marks_cache_;
};

}  // namespace autograd

#endif  // AUTOGRAD_TENSOR_H_