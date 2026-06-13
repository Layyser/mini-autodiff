#include "nn/optimizer.h"
#include <cmath>

namespace autograd {

// --- SGD Implementation ---
void SGD::step() {
  for (size_t i = 0; i < params_.size(); ++i) {
    auto& param = params_[i];
    if (param.grad().empty()) continue; // No grad to update

    std::vector<float>& data = param.mutable_data();
    const std::vector<float>& grad = param.grad();

    if (momentum_ > 0.0f) {
      std::vector<float>& vel = velocities_[i];
      for (size_t j = 0; j < data.size(); ++j) {
        // v = momentum * v - lr * grad
        // w = w + v
        vel[j] = momentum_ * vel[j] - lr_ * grad[j];
        data[j] += vel[j];
      }
    } else {
      // Standard SGD: w = w - lr * grad
      for (size_t j = 0; j < data.size(); ++j) {
        data[j] -= lr_ * grad[j];
      }
    }
  }
}

// --- Adam Implementation ---
void Adam::step() {
  t_++;
  for (size_t i = 0; i < params_.size(); ++i) {
    auto& param = params_[i];
    if (param.grad().empty()) continue;

    std::vector<float>& data = param.mutable_data();
    const std::vector<float>& grad = param.grad();
    std::vector<float>& m = m_[i];
    std::vector<float>& v = v_[i];

    // Corrections for bias
    float beta1_corr = 1.0f - std::pow(beta1_, t_);
    float beta2_corr = 1.0f - std::pow(beta2_, t_);

    for (size_t j = 0; j < data.size(); ++j) {
      float g = grad[j];

      // Update biased first moment estimate
      m[j] = beta1_ * m[j] + (1.0f - beta1_) * g;
      
      // Update biased second raw moment estimate
      v[j] = beta2_ * v[j] + (1.0f - beta2_) * (g * g);

      // Compute bias-corrected first moment estimate
      float m_hat = m[j] / beta1_corr;

      // Compute bias-corrected second raw moment estimate
      float v_hat = v[j] / beta2_corr;

      // Update parameters
      data[j] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
    }
  }
}

}  // namespace autograd