#include "value.h"
#include <algorithm> // for std::fill
#include <unordered_set>
#include <cassert>

namespace autograd {

// --- 1. The Internal Body (The "TV") ---
struct ValueImpl {
  float data;
  float grad;
  bool requires_grad;
  
  // Graph topology
  std::vector<std::shared_ptr<ValueImpl>> parents;
  std::function<void()> backward_fn;

  ValueImpl(float d, bool req_grad) 
  : data(d), grad(0.0f), requires_grad(req_grad) {}
};

// --- 2. Constructors & Accessors (The "Remote") ---
Value::Value() : node_(std::make_shared<ValueImpl>(0.0f, false)) {}

Value::Value(float data, bool requires_grad) 
    : node_(std::make_shared<ValueImpl>(data, requires_grad)) {}

Value::Value(std::shared_ptr<ValueImpl> node) : node_(std::move(node)) {}

float& Value::mutable_data() { return node_->data; }
float Value::data() const { return node_->data; }
float Value::grad() const { return node_->grad; }
float& Value::mutable_grad() { return node_->grad; }
bool Value::requires_grad() const { return node_->requires_grad; }

void Value::zero_grad() { node_->grad = 0.0f; }

// --- 3. The Backward Pass (Topological Sort) ---

static void BuildTopo(ValueImpl* node, 
                      std::vector<ValueImpl*>& topo, 
                      std::unordered_set<ValueImpl*>& visited) {
  if (visited.count(node)) return;
  visited.insert(node);

  for (auto& parent : node->parents) {
    BuildTopo(parent.get(), topo, visited);
  }
  topo.push_back(node);
}

void Value::backward() {
  if (!node_->requires_grad) return;

  // 1. Seed the gradient at the end of the chain
  node_->grad = 1.0f;

  // 2. Build the graph order
  std::vector<ValueImpl*> topo;
  std::unordered_set<ValueImpl*> visited;
  BuildTopo(node_.get(), topo, visited);

  // 3. Process in reverse order
  for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
    if ((*it)->backward_fn) {
      (*it)->backward_fn();
    }
  }
}

// --- 4. Operators ---

Value operator+(const Value& a, const Value& b) {
  float out_data = a.data() + b.data();
  bool req_grad = a.requires_grad() || b.requires_grad();
  
  auto out_node = std::make_shared<ValueImpl>(out_data, req_grad);
  
  if (req_grad) {
    out_node->parents = {a.node_, b.node_};
    
    // Capture parents by value (shared_ptr copy) to keep them alive
    auto node_a = a.node_;
    auto node_b = b.node_;
    
    // We use a raw pointer or weak_ptr for 'out_node' inside the lambda 
    // to avoid a circular reference (Out -> Lambda -> Out). 
    // But since 'out_node' owns the lambda, we can't capture 'out_node' shared_ptr.
    // We capture a weak_ptr to self.
    std::weak_ptr<ValueImpl> weak_out = out_node;

    out_node->backward_fn = [weak_out, node_a, node_b]() {
      auto out = weak_out.lock();
      if (!out) return;

      if (node_a->requires_grad) node_a->grad += 1.0f * out->grad;
      if (node_b->requires_grad) node_b->grad += 1.0f * out->grad;
    };
  }
  return Value(out_node);
}

Value operator-(const Value& a, const Value& b) {
  float out_data = a.data() - b.data();
  bool req_grad = a.requires_grad() || b.requires_grad();
  auto out_node = std::make_shared<ValueImpl>(out_data, req_grad);

  if (req_grad) {
    out_node->parents = {a.node_, b.node_};
    auto node_a = a.node_;
    auto node_b = b.node_;
    std::weak_ptr<ValueImpl> weak_out = out_node;

    out_node->backward_fn = [weak_out, node_a, node_b]() {
      auto out = weak_out.lock();
      if (!out) return;
      if (node_a->requires_grad) node_a->grad += 1.0f * out->grad;
      if (node_b->requires_grad) node_b->grad -= 1.0f * out->grad; // Note the minus
    };
  }
  return Value(out_node);
}

Value operator*(const Value& a, const Value& b) {
  float out_data = a.data() * b.data();
  bool req_grad = a.requires_grad() || b.requires_grad();
  auto out_node = std::make_shared<ValueImpl>(out_data, req_grad);

  if (req_grad) {
    out_node->parents = {a.node_, b.node_};
    auto node_a = a.node_;
    auto node_b = b.node_;
    std::weak_ptr<ValueImpl> weak_out = out_node;

    out_node->backward_fn = [weak_out, node_a, node_b]() {
      auto out = weak_out.lock();
      if (!out) return;
      
      // Product Rule: d(ab)/da = b, d(ab)/db = a
      if (node_a->requires_grad) node_a->grad += node_b->data * out->grad;
      if (node_b->requires_grad) node_b->grad += node_a->data * out->grad;
    };
  }
  return Value(out_node);
}

Value operator/(const Value& a, const Value& b) {
  float out_data = a.data() / b.data();
  bool req_grad = a.requires_grad() || b.requires_grad();
  auto out_node = std::make_shared<ValueImpl>(out_data, req_grad);

  if (req_grad) {
    out_node->parents = {a.node_, b.node_};
    auto node_a = a.node_;
    auto node_b = b.node_;
    std::weak_ptr<ValueImpl> weak_out = out_node;

    out_node->backward_fn = [weak_out, node_a, node_b]() {
      auto out = weak_out.lock();
      if (!out) return;

      // Quotient Rule
      if (node_a->requires_grad) node_a->grad += (1.0f / node_b->data) * out->grad;
      if (node_b->requires_grad) node_b->grad -= (node_a->data / (node_b->data * node_b->data)) * out->grad;
    };
  }
  return Value(out_node);
}

}  // namespace autograd