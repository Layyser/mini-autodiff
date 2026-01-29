#include <iostream>
#include <vector>
#include "value.h"

using autograd::Value;

// The Sanity Check from the previous chapter
void DocsExample() {
  std::cout << "--- Sanity Check ---\n";
  // Example: f(a, b) = (a + b) * b
  // Let a = 2.0, b = 3.0
  // f = (2 + 3) * 3 = 15
  
  // Analytical Derivatives:
  // df/da = b = 3
  // df/db = (a + b) * 1 + b * 1 = (2 + 3) + 3 = 8
  
  Value a(2.0f, true);
  Value b(3.0f, true);
  
  Value c = a + b;  // 5.0
  Value f = c * b;  // 15.0
  
  f.backward();
  
  std::cout << "Result: " << f.data() << " (Expected 15.0)" << std::endl;
  std::cout << "Grad a: " << a.grad() << " (Expected 3.0)" << std::endl;
  std::cout << "Grad b: " << b.grad() << " (Expected 8.0)" << std::endl;
  std::cout << "--------------------\n\n";
}

// The Training Loop
void TrainLinearModel() {
  std::cout << "--- Training Linear Model (Target: y = 3x + 2) ---\n";

  // 1. Dataset (y = 3x + 2)
  // x: 0, 1, 2, 3
  // y: 2, 5, 8, 11
  std::vector<float> X = {0.0f, 1.0f, 2.0f, 3.0f};
  std::vector<float> Y = {2.0f, 5.0f, 8.0f, 11.0f};

  // 2. Initialize Parameters
  // We want the model to learn w=3, b=2. Start them at random/zero.
  Value w(0.5f, true); // Random guess
  Value b(0.0f, true); // Random guess

  float learning_rate = 0.01f;

  // 3. Training Loop
  for (int epoch = 0; epoch < 200; ++epoch) {
    Value total_loss(0.0f);

    // Accumulate loss over the batch
    for (size_t i = 0; i < X.size(); ++i) {
      Value x_val(X[i]);
      Value y_target(Y[i]);

      // Forward Pass: y = wx + b
      Value prediction = (x_val * w) + b;

      // Loss: (prediction - target)^2
      Value diff = prediction - y_target;
      Value loss = diff * diff;

      total_loss = total_loss + loss;
    }

    // Zero Gradients before backward!
    // (In C++, we must manually clear them or they accumulate)
    w.zero_grad();
    b.zero_grad();

    // Backward Pass
    total_loss.backward();

    // Optimizer Step (Gradient Descent)
    // w = w - learning_rate * grad
    // We modify data directly to avoid adding this math to the graph
    w.mutable_data() -= learning_rate * w.grad();
    b.mutable_data() -= learning_rate * b.grad();

    if (epoch % 10 == 0) {
      std::cout << "Epoch " << epoch << " | Loss: " << total_loss.data() 
                << " | w: " << w.data() << " | b: " << b.data() << std::endl;
    }
  }

  std::cout << "Final Results -> w: " << w.data() << " (Exp: 3.0), b: " << b.data() << " (Exp: 2.0)\n";
}

int main() {
  DocsExample();
  TrainLinearModel();
  return 0;
}