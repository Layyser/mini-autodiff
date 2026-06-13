#ifndef AUTOGRAD_LOSS_H_
#define AUTOGRAD_LOSS_H_

#include "core/tensor.h"
#include <vector>

namespace autograd {

// Abstract Base Class
class Loss {
 public:
  virtual ~Loss() = default;
  virtual Tensor operator()(const Tensor& pred, const Tensor& target) = 0;
};

// --- Mean Squared Error ---
class MSELoss : public Loss {
 public:
  Tensor operator()(const Tensor& pred, const Tensor& target) override {
    Tensor diff = pred - target;
    return mean(diff * diff);
  }
};

// --- Binary Cross Entropy ---
class BCELoss : public Loss {
 public:
  Tensor operator()(const Tensor& pred, const Tensor& target) override {
    // Clamp away from 0/1: log(pred) and log(1-pred) would otherwise
    // produce -inf/NaN when pred saturates (common with sigmoid + float32).
    constexpr float kEps = 1e-7f;
    Tensor clamped_pred = clamp(pred, kEps, 1.0f - kEps);

    Tensor one({1.0f}, {1}, false);
    Tensor neg_one({-1.0f}, {1}, false);

    Tensor term1 = target * log(clamped_pred);
    Tensor term2 = (one - target) * log(one - clamped_pred);

    return mean(term1 + term2) * neg_one;
  }
};

}  // namespace autograd

#endif  // AUTOGRAD_LOSS_H_