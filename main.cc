#include <iostream>
#include <vector>
using std::vector, std::cout;

// Tensor class
#include "tensor.h"
using autograd::Tensor;

// Optimizers
#include "optimizer.h"
using autograd::SGD, autograd::Adam;

// Loss function
#include "loss.h"
using autograd::MSELoss;


void TrainLinearRegression() {
  // --- Data Setup ---
  Tensor X({0.0f, 1.0f, 2.0f, 3.0f, 4.0f}, {5, 1});
  Tensor Y({2.0f, 5.0f, 8.0f, 11.0f, 14.0f}, {5, 1});

  // --- Weights & Bias separated ---
  Tensor w({0.0f}, {1, 1}, true); 
  Tensor b({0.0f}, {1}, true);

  // --- Defining criterion ---
  MSELoss criterion;

  // --- Optimizer ---
  float LR = 0.001f;
  Adam optimizer({w, b}, LR);

  // --- Training Loop ---
  int EPOCHS = 50000;
  for (int epoch = 0; epoch <= EPOCHS; ++epoch) {
    optimizer.zero_grad();
    Tensor pred = matmul(X, w) + b; // Linear model
    Tensor loss = criterion(pred, Y);
    loss.backward();

    // --- Optimization ---
    optimizer.step();

    if (epoch % 100 == 0) {
      cout << "Epoch " << epoch << " Loss: " << loss.data()[0]
           << " | W: " << w.data()[0] << " B: " << b.data()[0] << '\n';
    }
  }
}


void TrainMLP() {
  // --- XOR Data (Classic non-linear test) ---
  // 4 samples, 2 features
  Tensor X({0.0f, 0.0f, 
            0.0f, 1.0f, 
            1.0f, 0.0f, 
            1.0f, 1.0f}, {4, 2});
  
  // Target: XOR output
  Tensor Y({0.0f, 
            1.0f, 
            1.0f, 
            0.0f}, {4, 1});

  // --- MLP Architecture ---
  // Hidden Layer: 2 inputs -> 4 hidden units
  Tensor W1 = Tensor({ // Random initialization (simplified for demo)
      0.1f, -0.2f, 0.5f, 0.1f,
      -0.4f, 0.3f, 0.2f, -0.5f
  }, {2, 4}, true);
  // Bias for Hidden Layer: Shape {4} (Will be broadcasted to {4, 4})
  Tensor b1 = Tensor({0.1f, 0.0f, -0.1f, 0.0f}, {4}, true); 

  // Output Layer: 4 hidden -> 1 output
  Tensor W2 = Tensor({0.2f, -0.5f, 0.1f, 0.4f}, {4, 1}, true);
  Tensor b2 = Tensor({0.0f}, {1}, true);

  Adam optimizer({W1, b1, W2, b2}, 0.01f);
  MSELoss criterion;

  cout << "Training MLP on XOR...\n";

  for (int epoch = 0; epoch < 1000; ++epoch) {
    optimizer.zero_grad();

    // Forward Pass
    // 1. Hidden Layer
    Tensor h_pre = matmul(X, W1) + b1; // Broadcasting b1 here!
    Tensor h = relu(h_pre); 
    
    // 2. Output Layer
    Tensor out = matmul(h, W2) + b2;   // Broadcasting b2 here!
    Tensor loss = criterion(out, Y);

    loss.backward();
    optimizer.step();

    if (epoch % 100 == 0) {
        cout << "Epoch " << epoch << " Loss: " << loss.data()[0] << "\n";
    }
  }
  
  // Final Prediction
  Tensor final_pred = matmul(X, W1) + b1;
  final_pred = relu(final_pred);
  final_pred = matmul(final_pred, W2) + b2;
  
  cout << "\nFinal Predictions:\n";
  for(float val : final_pred.data()) {
    cout << val << " ";
  }
  cout << "\n";

}

int main() {
  TrainMLP();
  return 0;
}