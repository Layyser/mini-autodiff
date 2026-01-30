#include <iostream>
#include <vector>
#include "tensor.h"

using autograd::Tensor;
using std::vector, std::cout;

int main() {
  // --- Data Setup ---
  vector<float> X_raw = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  vector<float> Y_raw = {2.0f, 5.0f, 8.0f, 11.0f, 14.0f};
  Tensor X({X_raw}, {5, 1});
  Tensor Y({Y_raw}, {5, 1});

  // --- Weights & Bias separated ---
  // Weight: Shape {1, 1} (Scalar matrix)
  Tensor w({0.0f}, {1, 1}, true); 
  
  // Bias: Shape {1} (Scalar vector)
  Tensor b({0.0f}, {1}, true);

  float LR = 0.0001f;
  float EPOCHS = 200000;
  for (int epoch = 0; epoch <= EPOCHS; ++epoch) {
    // --- Autograd API ---
    b.zero_grad();
    w.zero_grad();

    float total_loss = 0.0f;
    Tensor pred = matmul(X, w) + b;
    Tensor diff = pred - Y;

    // MSE formula
    Tensor losses = diff * diff;
    Tensor loss = mean(losses);

    loss.backward();
    total_loss += loss.data()[0];

    // --- Optimization ---
    w.mutable_data()[0] -= LR * w.grad()[0];
    b.mutable_data()[0] -= LR * b.grad()[0];

    if (epoch % 100 == 0) {
      cout << "Epoch " << epoch << " Loss: " << total_loss
           << " | W: " << w.data()[0] << " B: " << b.data()[0] << '\n';
    }
  }
  
  return 0;
}