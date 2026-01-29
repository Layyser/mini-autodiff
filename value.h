#ifndef AUTOGRAD_VALUE_H_
#define AUTOGRAD_VALUE_H_

#include <functional>
#include <memory>
#include <vector>
#include <string>

namespace autograd {

// Forward declaration of the internal body
struct ValueImpl;

class Value {
 public:
  // Constructors
  Value();
  Value(float data, bool requires_grad = false);

  // Accessors
  
  float data() const;
  float& mutable_data();
  float grad() const;
  float& mutable_grad();
  bool requires_grad() const;
  
  // Zeros out the gradient (useful for optimization loops)
  void zero_grad();

  // The engine trigger
  void backward();

  // Operators
  friend Value operator+(const Value& a, const Value& b);
  friend Value operator-(const Value& a, const Value& b);
  friend Value operator*(const Value& a, const Value& b);
  friend Value operator/(const Value& a, const Value& b);

 private:
  // Private constructor for internal graph building
  explicit Value(std::shared_ptr<ValueImpl> node);

  std::shared_ptr<ValueImpl> node_;
};

}  // namespace autograd

#endif  // AUTOGRAD_VALUE_H_