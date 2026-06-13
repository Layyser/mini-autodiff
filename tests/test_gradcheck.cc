// Gradient-checking test harness.
//
// For each op, computes analytic gradients via Tensor::backward() and
// compares them against numerical gradients obtained via central
// differences. Also covers the matmul shape-validation added alongside
// this harness.

#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/tensor.h"
#include "nn/loss.h"

using autograd::BCELoss, autograd::MSELoss, autograd::Tensor;
using std::cerr, std::cout, std::string, std::vector;

// matmul, log, sum, mean, relu, sigmoid are hidden friends of Tensor,
// found via ADL on their Tensor arguments below.

namespace {

int g_failures = 0;

// Estimate d f / d inputs[idx][elem] via central differences.
float NumericalGrad(const std::function<float(vector<Tensor>&)>& f,
                     vector<Tensor>& inputs, size_t idx, size_t elem,
                     float eps = 1e-3f) {
  auto& data = inputs[idx].mutable_data();
  float orig = data[elem];

  data[elem] = orig + eps;
  float plus = f(inputs);

  data[elem] = orig - eps;
  float minus = f(inputs);

  data[elem] = orig;
  return (plus - minus) / (2.0f * eps);
}

// Runs forward + backward through `f` and compares the analytic gradients
// against numerical gradients for every element of every input tensor.
void GradCheck(const string& name,
               const std::function<Tensor(vector<Tensor>&)>& f,
               vector<Tensor> inputs, float tol = 1e-2f) {
  Tensor out = f(inputs);
  out.backward();

  auto scalar_f = [&](vector<Tensor>& xs) -> float { return f(xs).data()[0]; };

  for (size_t i = 0; i < inputs.size(); ++i) {
    if (!inputs[i].requires_grad()) continue;
    const auto& analytic = inputs[i].grad();

    for (int j = 0; j < inputs[i].size(); ++j) {
      float numeric = NumericalGrad(scalar_f, inputs, i, j);
      float diff = std::fabs(numeric - analytic[j]);

      if (diff > tol) {
        cerr << "[FAIL] " << name << ": input " << i << " elem " << j
             << " analytic=" << analytic[j] << " numeric=" << numeric
             << " diff=" << diff << "\n";
        ++g_failures;
        return;
      }
    }
  }
  cout << "[PASS] " << name << "\n";
}

// --- Op coverage ---

void TestAdd() {
  GradCheck("Add (same shape)",
            [](vector<Tensor>& xs) { return sum(xs[0] + xs[1]); },
            {Tensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, true),
             Tensor({0.5f, -1.0f, 2.0f, 3.0f}, {2, 2}, true)});

  GradCheck("Add (broadcast)",
            [](vector<Tensor>& xs) { return sum(xs[0] + xs[1]); },
            {Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, true),
             Tensor({0.1f, 0.2f, 0.3f}, {1, 3}, true)});
}

void TestSub() {
  GradCheck("Sub (same shape)",
            [](vector<Tensor>& xs) { return sum(xs[0] - xs[1]); },
            {Tensor({3.0f, 1.0f, -2.0f, 5.0f}, {2, 2}, true),
             Tensor({1.0f, 1.0f, 1.0f, 1.0f}, {2, 2}, true)});

  GradCheck("Sub (broadcast)",
            [](vector<Tensor>& xs) { return sum(xs[0] - xs[1]); },
            {Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, true),
             Tensor({0.1f, 0.2f, 0.3f}, {3}, true)});
}

void TestMul() {
  GradCheck("Mul (same shape)",
            [](vector<Tensor>& xs) { return sum(xs[0] * xs[1]); },
            {Tensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, true),
             Tensor({0.5f, -1.5f, 2.5f, -3.5f}, {2, 2}, true)});

  GradCheck("Mul (broadcast)",
            [](vector<Tensor>& xs) { return sum(xs[0] * xs[1]); },
            {Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, true),
             Tensor({1.5f, -2.0f, 0.5f}, {1, 3}, true)});
}

void TestDiv() {
  // Denominators kept away from zero.
  GradCheck("Div (same shape)",
            [](vector<Tensor>& xs) { return sum(xs[0] / xs[1]); },
            {Tensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, true),
             Tensor({2.0f, -4.0f, 1.5f, 5.0f}, {2, 2}, true)});

  GradCheck("Div (broadcast)",
            [](vector<Tensor>& xs) { return sum(xs[0] / xs[1]); },
            {Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, true),
             Tensor({2.0f, -3.0f, 4.0f}, {1, 3}, true)});
}

void TestMatMul() {
  GradCheck(
      "MatMul",
      [](vector<Tensor>& xs) { return sum(matmul(xs[0], xs[1])); },
      {Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, true),
       Tensor({1.0f, -1.0f, 0.5f, 2.0f, -0.5f, 1.0f}, {3, 2}, true)});

  // Mismatched inner dimensions must raise, not read out of bounds.
  bool threw = false;
  try {
    Tensor a({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    Tensor b({1.0f, 2.0f, 3.0f}, {3, 1});
    matmul(a, b);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  if (threw) {
    cout << "[PASS] MatMul (rejects mismatched inner dimensions)\n";
  } else {
    cerr << "[FAIL] MatMul (rejects mismatched inner dimensions): no exception thrown\n";
    ++g_failures;
  }

  // Non-2D operands must raise, not read out of bounds.
  threw = false;
  try {
    Tensor a({1.0f, 2.0f, 3.0f}, {3});
    Tensor b({1.0f, 2.0f, 3.0f}, {3});
    matmul(a, b);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  if (threw) {
    cout << "[PASS] MatMul (rejects non-2D operands)\n";
  } else {
    cerr << "[FAIL] MatMul (rejects non-2D operands): no exception thrown\n";
    ++g_failures;
  }
}

void TestNeg() {
  GradCheck("Neg", [](vector<Tensor>& xs) { return sum(-xs[0]); },
            {Tensor({1.0f, -2.0f, 3.0f, -4.0f}, {2, 2}, true)});
}

void TestRelu() {
  // Values kept away from the kink at zero.
  GradCheck("Relu", [](vector<Tensor>& xs) { return sum(relu(xs[0])); },
            {Tensor({1.5f, -2.0f, 3.5f, -0.5f}, {2, 2}, true)});
}

void TestSigmoid() {
  GradCheck("Sigmoid", [](vector<Tensor>& xs) { return sum(sigmoid(xs[0])); },
            {Tensor({0.5f, -1.0f, 2.0f, -2.5f}, {2, 2}, true)});
}

void TestLog() {
  // Strictly positive inputs.
  GradCheck("Log", [](vector<Tensor>& xs) { return sum(log(xs[0])); },
            {Tensor({0.5f, 1.0f, 2.0f, 3.5f}, {2, 2}, true)});
}

void TestClamp() {
  // Values kept away from the [-1, 1] boundaries so finite differences
  // don't straddle the kink.
  GradCheck("Clamp",
            [](vector<Tensor>& xs) { return sum(clamp(xs[0], -1.0f, 1.0f)); },
            {Tensor({-2.0f, -0.5f, 0.5f, 2.0f}, {2, 2}, true)});
}

void TestSum() {
  GradCheck("Sum", [](vector<Tensor>& xs) { return sum(xs[0]); },
            {Tensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, true)});
}

void TestMean() {
  GradCheck("Mean", [](vector<Tensor>& xs) { return mean(xs[0]); },
            {Tensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, true)});
}

// --- Loss coverage ---

void TestMSELoss() {
  MSELoss criterion;
  GradCheck("MSELoss",
            [&](vector<Tensor>& xs) { return criterion(xs[0], xs[1]); },
            {Tensor({0.8f, 0.2f, 0.6f, 0.4f}, {2, 2}, true),
             Tensor({1.0f, 0.0f, 1.0f, 0.0f}, {2, 2}, false)});
}

void TestBCELoss() {
  BCELoss criterion;
  // Predictions kept away from 0/1 where log() is unstable.
  GradCheck("BCELoss",
            [&](vector<Tensor>& xs) { return criterion(xs[0], xs[1]); },
            {Tensor({0.7f, 0.3f, 0.6f, 0.4f}, {2, 2}, true),
             Tensor({1.0f, 0.0f, 1.0f, 0.0f}, {2, 2}, false)});
}

// Perfectly-confident (saturated) predictions used to produce NaN via
// 0 * log(0) before BCELoss clamped its input.
void TestBCELossStability() {
  BCELoss criterion;
  Tensor pred({0.0f, 1.0f, 0.0f, 1.0f}, {2, 2}, true);
  Tensor target({0.0f, 1.0f, 0.0f, 1.0f}, {2, 2}, false);

  Tensor loss = criterion(pred, target);
  float value = loss.data()[0];

  if (std::isnan(value) || std::isinf(value)) {
    cerr << "[FAIL] BCELoss (stability): loss is " << value
         << " for saturated predictions\n";
    ++g_failures;
    return;
  }
  cout << "[PASS] BCELoss (stability)\n";
}

// --- Engine API checks ---

// backward() requires a scalar (size 1) output.
void TestBackwardRequiresScalar() {
  Tensor x({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, true);
  Tensor y = x + x;  // size 4, not a scalar

  bool threw = false;
  try {
    y.backward();
  } catch (const std::runtime_error&) {
    threw = true;
  }

  if (threw) {
    cout << "[PASS] backward() rejects non-scalar output\n";
  } else {
    cerr << "[FAIL] backward() rejects non-scalar output: no exception thrown\n";
    ++g_failures;
  }
}

}  // namespace

int main() {
  TestAdd();
  TestSub();
  TestMul();
  TestDiv();
  TestMatMul();
  TestNeg();
  TestRelu();
  TestSigmoid();
  TestLog();
  TestClamp();
  TestSum();
  TestMean();
  TestMSELoss();
  TestBCELoss();
  TestBCELossStability();
  TestBackwardRequiresScalar();

  if (g_failures > 0) {
    cerr << "\n" << g_failures << " test(s) failed.\n";
    return 1;
  }
  cout << "\nAll tests passed.\n";
  return 0;
}
