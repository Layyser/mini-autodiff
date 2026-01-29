#include <iostream>
#include <vector>
#include <iomanip>
#include "tensor.h"

using autograd::Tensor;

int main() {
  // --- Data Setup ---
  // We want to learn y = 3x + 2.
  // We format input X as [x_val, 1.0] so we can use dot product.
  std::vector<std::vector<float>> X_data = {
    {0.0f, 1.0f}, {1.0f, 1.0f}, {2.0f, 1.0f}, {3.0f, 1.0f}, {4.0f, 1.0f}
  };
  std::vector<float> Y_data = {2.0f, 5.0f, 8.0f, 11.0f, 14.0f};

  // --- Weights ---
  // CHANGE 1: Explicit shape {2, 1} (Column vector)
  // This allows matmul: (1x2) * (2x1) = (1x1)
  Tensor weights({0.0f, 0.0f}, {2, 1}, true); 

  float learning_rate = 0.001f;

  std::cout << "Target: [3.0, 2.0]\n";

  for (int epoch = 0; epoch < 2000; ++epoch) {
    float total_loss = 0.0f;

    for (size_t i = 0; i < X_data.size(); ++i) {
      Tensor x(X_data[i], {1, 2}, false); 
      Tensor y({Y_data[i]}, {1, 1}, false); 

      // Prediction: 
      // [x, 1] . [w, b]
      // [1, 2] . [2, 1] -> [1, 1]
      Tensor pred = matmul(x, weights); 

      Tensor diff = pred - y;
      Tensor loss = diff * diff; // MSE loss

      loss.backward();
      total_loss += loss.data()[0];
    }

    // --- Optimization ---
    float N = (float)X_data.size();
    std::vector<float> new_w_data = weights.data();
    
    // Update [w, b]
    for(size_t i = 0; i < new_w_data.size(); ++i) {
      new_w_data[i] -= learning_rate * (weights.grad()[i] / N); 
    }
    
    weights = Tensor(new_w_data, {2, 1}, true);

    if (epoch % 100 == 0) {
      std::cout << "Epoch " << epoch << " Loss: " << (total_loss/N) 
                << " | Weights: [" << weights.data()[0] << ", " << weights.data()[1] << "]\n";
    }
  }
  
  return 0;
}