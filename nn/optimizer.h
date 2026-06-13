#ifndef AUTOGRAD_OPTIMIZER_H_
#define AUTOGRAD_OPTIMIZER_H_

#include "core/tensor.h"
#include <vector>

namespace autograd {

class Optimizer {
 public:
  Optimizer(std::vector<Tensor> params, float lr) 
      : params_(std::move(params)), lr_(lr) {}
  
  virtual ~Optimizer() = default;

  // Resets gradients of all registered parameters to zero
  void zero_grad() {
    for (auto& p : params_) {
      p.zero_grad();
    }
  }

  // Performs a single optimization step (updates parameters)
  virtual void step() = 0;

 protected:
  std::vector<Tensor> params_;
  float lr_;
};

// --- Stochastic Gradient Descent ---
class SGD : public Optimizer {
 public:
  SGD(std::vector<Tensor> params, float lr, float momentum = 0.0f)
      : Optimizer(std::move(params), lr), momentum_(momentum) {
    if (momentum_ > 0.0f) {
      velocities_.resize(params_.size());
      for (size_t i = 0; i < params_.size(); ++i) {
        velocities_[i] = std::vector<float>(params_[i].size(), 0.0f);
      }
    }
  }

  void step() override;

 private:
  float momentum_;
  std::vector<std::vector<float>> velocities_;
};

// --- Adam (Adaptive Moment Estimation) ---
class Adam : public Optimizer {
 public:
  Adam(std::vector<Tensor> params, float lr, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f)
      : Optimizer(std::move(params), lr), beta1_(beta1), beta2_(beta2), eps_(eps), t_(0) {
    // Initialize moment buffers
    m_.resize(params_.size());
    v_.resize(params_.size());
    for (size_t i = 0; i < params_.size(); ++i) {
      size_t size = params_[i].size();
      m_[i] = std::vector<float>(size, 0.0f);
      v_[i] = std::vector<float>(size, 0.0f);
    }
  }

  void step() override;

 private:
  float beta1_;
  float beta2_;
  float eps_;
  int t_; // Time step
  std::vector<std::vector<float>> m_; // First moment
  std::vector<std::vector<float>> v_; // Second moment
};

}  // namespace autograd

#endif  // AUTOGRAD_OPTIMIZER_H_