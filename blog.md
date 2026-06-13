# Introduction
Have you ever wondered how PyTorch, TensorFlow, or JAX actually make Machine Learning work at a low level? Or maybe you have never used any of these frameworks and simply want a deep and clear introduction to Machine Learning itself.

This blog documents my journey into writing an automatic differentiation engine in C++ completely from scratch, while covering the fundamentals of Machine Learning and its underlying mathematics in a beginner-friendly way.

A basic understanding of graph theory and data structures will be helpful. We will move step-by-step, but if you feel lost at any point, feel free to pause and review the core concepts before moving forward.

**Disclaimer:** This engine is not optimized yet, and that is completely intentional. The goal here is to keep things simple, didactic, and educational. Whenever a design choice is made for clarity rather than performance, I will explicitly say so. If you are already experienced in the field, feel free to provide feedback or discuss state-of-the-art implementations. This project will eventually be optimized and rewritten for CUDA in the future.


## AI Prelude: Hill Climbing
Before diving into automatic differentiation, it is important to understand why it matters in the first place. If you are already familiar with optimization problems, [you may skip this section](#the-calculus-of-improvement), but I strongly recommend at least giving it a quick look to build intuition.

To understand modern Machine Learning, it helps to start historically, around the 1950s and 1960s, when early AI systems were mainly based on search and optimization techniques. One of the most influential ideas of that era was Hill Climbing.

From a very general point of view, we can describe Machine Learning as the process of finding a solution ($S$) to a given problem ($P$). In abstract terms, this is simply a function:$$f : P \to S$$

However, we must make an important distinction. In the simplest setting, a model behaves like a "black box"; we input data, get an output, and don't care how the decision was made.
For example:
- An image goes in, a label comes out. (e.g., Does this image contain a cat? Y/N)
- A set of numbers goes in, a prediction comes out. (e.g., What will the temperature be like tomorrow given the last 3 days temperatures? 15,3°C)

But optimization algorithms (like Hill Climbing) do something more subtle. Instead of just mapping input directly to output, they engage in a process of improvement. They start with a bad guess and iteratively refine it until it becomes a good solution. In optimization and learning, this usually takes the form of a structured exploration process.


### The Hiking Analogy
Before introducing any mathematics, let us first think at a very high level. Imagine we are standing somewhere in a vast terrain and our goal is to reach the top of the highest hill. We do not know in advance where the peak is, nor what the full shape of the landscape looks like. We can only observe our immediate surroundings.

1. At any moment, we can perform a small set of allowed moves: take a step north, south, east, or west.

2. To decide where to go next, we use a simple intuition: How high am I right now? And in which direction does the terrain seem to rise the most?

3. We then choose the move that increases our altitude the most. Step by step, without ever seeing the whole map, we gradually climb uphill. This is, in essence, what optimization algorithms do.


### Formalizing the Idea
We first define what a state is. Formally, a state $s \in S$ belongs to a state space $S$ and represents the current configuration of the system.

1. A state may encode the current parameters of a model $\theta \in \mathbb{R}^n$, a position in a search space, or a partial construction of a solution.

2. Next, we define a set of operators (or actions) $A = \{a_1, a_2, ..., a_k\}$, where each operator is a transformation $a : S \to S$ that maps one state to another.

3. Finally, we introduce a heuristic or objective function $h : S \to \mathbb{R}$, which assigns a scalar score to each state, measuring how "good" that state is. At each step $t$, given the current state, the algorithm selects the operator that maximizes improvement:$$a_t = \text{argmax}_{a \in A} \ h(a(s_t))$$

In other words, instead of directly computing a solution, we construct a sequence of states:$$s_0 \to s_1 \to s_2 \to ... \to s_T$$such that the heuristic value $h(s_t)$ monotonically improves along the trajectory.


### The Limitations of Hill Climbing

While intuitive, this approach highlights three major problems that modern Machine Learning had to solve:

#### 1. The Differentiation Problem (Which way is up?)  
Hill Climbing requires us to "try" every possible move (North, South, East, West) to see which one is better. In a low-dimensional space, checking 4 directions is fine. But imagine a neural network with 1 billion parameters. We cannot possibly "check" every direction. 

#### 2. The "Human in the Loop" Problem (Where does $h$ come from?)
In classical AI, the heuristic $h$ was often designed by humans. We had to manually tell the computer what looked like a good state (e.g., "in chess, losing a queen is bad"). But what if the problem is too complex for us to define $h$ manually? Or in general, what if we can't find a good enough heuristic? In modern Deep Learning, instead of hand-designing heuristics that directly evaluate states, we define a generic loss function and let the system learn the parameters of a model that implicitly shapes the landscape.

#### 3. The Local Maximum Problem (What if we get stuck?)
Hill Climbing is "greedy." It always takes the step that improves the score right now. But sometimes, to reach the highest peak (Global Maximum), you first have to go down into a valley. If you blindly follow the steepest path, you might end up on a small hill (Local Maximum) and get stuck there, seeing no way up, even though the true peak is kilometers away.

// TODO add Image of Local Maximum with h(x) vs x


It is clear that to solve the first problem, we need a mathematical way to instantly know without trial and error exactly which direction leads uphill. This is where calculus and automatic differentiation enter the picture.


## The Calculus of Improvement

Whether we are climbing simple hills or training massive neural networks, we eventually face a critical bottleneck: **The Differentiation Problem**.

If our model has millions of parameters, we cannot afford to blindly test every direction to see which one improves the result. We need a way to calculate the slope (gradient) of the landscape instantly, without taking a single step.  

In mathematics, the tool for finding the slope of a function is the **Derivative**.  

If we had a simple heuristic function like $h(x) = x^2$, we know from high school calculus that the derivative is $h'(x) = 2x$. This tells us exactly how $h$ changes as $x$ changes. But in Machine Learning, we aren't dealing with simple textbook functions. We are dealing with massive, composite functions made of millions of small operations chained together.  

To solve this efficiently, we rely on a technique called Reverse-Mode Automatic Differentiation, famously known in Deep Learning as Backpropagation.  

### Modeling the functions
To automate calculus, we must stop thinking of functions as abstract formulas and start thinking of them as graphs. Any complex calculation can be broken down into a sequence of elementary operations (addition, multiplication, power, etc.). If we represent these operations as nodes in a graph, we get a clear map of how data flows from input to output. Consider the expression:$$f(x, y) = (x + y) \cdot z$$  

We can decompose this into two steps: 
- Intermediate value $a = x + y$ 
- Output value $f = a \cdot z$

By structuring the math this way, we gain a superpower: **local knowledge**.  

In this picture you can observe how any function can be built as the composition of different functions, creating a Directed Acyclic Graph (DAG).

// TODO: Add Graph

Every node in this graph only needs to know its immediate neighbors to calculate its own derivative. The multiplication node ( $\cdot$ ) knows how to derive multiplication. The addition node ( $+$ ) knows how to derive addition. We don't need a super-genius algorithm that understands the whole complex function at once. We just need small, simple logical units that know how to pass gradient information to their parents.



### The Chain Rule 

This is where the magic happens. To know how the final output $f(x,y)$ changes with respect to the initial input $x$, we simply multiply the local derivatives along the path from $f$ backward to $x$. This is the Chain Rule in our previous function: 
$$\frac{\partial f}{\partial x} = \frac{\partial f}{\partial a} \cdot \frac{\partial a}{\partial x}$$

Essentially, we are breaking down the total influence into a chain of local influences. The way $x$ impacts the final result $f$ is determined by two things:
1. How strongly $x$ pushes $a$ (the **local derivative** of addition, remember: $a = x + y$ ).
2. How strongly $a$ pushes $f$ (the **local derivative** of multiplication, remember: $f = a \cdot z$).  

Because these effects are sequential, they compound. To find the total sensitivity of $f$ to $x$, we multiply these local sensitivities together. If changing $x$ by a tiny amount shifts $a$ by a factor of 2, and changing $a$ shifts $f$ by a factor of 3, then changing $x$ must ultimately shift $f$ by a factor of $2 \cdot 3 = 6$.

In our engine, "backpropagation" is simply the process of traversing this graph in reverse order. We start at the end $f$, set its gradient to 1, and then walk backward, asking every node to distribute the gradient to its children.


### The Objective
Why are we doing this? At this point, you might be wondering: "*I understand how to find the derivative of a math expression, but how does this equate to learning?*"

Remember the **Hill Climbing** analogy? We wanted to maximize a score; the heuristic function because we knew it meant it was better. In modern Machine Learning, we usually frame this in reverse: we want to minimize an error and we call this **Gradient Descent**.  

Hill Climbing estimates the slope by trying neighboring states. Gradient Descent computes the slope analytically, using calculus. Both search uphill, but one does it by probing, the other by measuring.

To solve our problem $f: P \to S$, we don't just calculate an output. We compare that output to a known "correct" answer (Ground Truth). We add one final node to our graph: the Loss Node ($L$). 
- This node calculates the difference between what our model predicted and what we actually wanted. 
- If the Loss is high, our model is bad. If the Loss is low, our model is good.

By applying the Chain Rule from this Loss node backwards, we get the gradient $\frac{\partial L}{\partial w}$ for every parameter $w$ in our system. This gradient tells us exactly how to tweak each parameter to slightly decrease the error. We aren't just calculating derivatives for fun; we are building a mechanism to mechanically reduce its own error.

### Why Reverse Mode?

At this point, a natural question arises: "*Why do we traverse the graph backward instead of forward?*"  
After all, derivatives can also be computed by propagating information from inputs to outputs.

The answer lies in efficiency. In Machine Learning, we typically have many inputs (millions of parameters) but only one final scalar output: the loss. Our goal is to compute the gradient of this single loss with respect to every parameter.

If we were to propagate derivatives forward (a technique known as Forward-Mode Automatic Differentiation), we would need to run a separate pass for each input variable. For a model with a billion parameters, that would mean a billion passes; completely infeasible.

Reverse-Mode Automatic Differentiation flips the perspective. Instead of asking “How does each input affect all outputs?”, we ask “How does the single output affect every input?” In one backward traversal of the graph, we obtain the gradient of the loss with respect to all parameters at once.

This is why backpropagation is the engine behind modern Deep Learning: when the number of parameters is huge and the output is a single scalar, reverse mode is not just convenient, it is the only practical choice.


## The Implementation
### Why C++?
You might be wondering: "*Why build this in C++? Isn't Machine Learning done in Python?*"

It is true that Python is the interface of modern AI, but C++ is the engine. Libraries like PyTorch and TensorFlow are essentially high-performance C++ systems wrapped in Python bindings. By stepping down into C++, we gain two massive advantages:

1. **No Magic**: In Python, memory management and object overhead are hidden. In C++, we have to explicitly design how nodes in our graph live and die. This forces us to understand the lifecycle of a gradient in a way Python never requires.

2. **Performance Potential**: While our starting scalar engine won't be faster than PyTorch, the architecture we build here paves the way for the high-performance, multithreaded tensor operations we will implement later.

If you understand how to build this in C++, you don't just understand the math; you understand the system.

## Version 0: The simplest engine
***Quick note**: You can check every version at the git history of commits*

### What do we need to build?
Before we write a single line of code, let's look at the ingredients required to bake an Autodiff engine. We don't need complex external libraries; we just need four core concepts:

1. **The Value Class:**
We need a class that acts like a number (holds a float) but carries extra baggage: a gradient and a list of parents.

2. **Operator Overloading:**
We need to hijack standard math symbols ($+$, $*$, $/$, ...). When we write `a + b`, our code shouldn't just calculate the sum; it must record the relationship between `a`, `b`, and the result.

3. **The Graph:**
Every time we perform an operation, we are secretly building the Directed Acyclic Graph (DAG) in memory. As seen in the previous chapter, this graph records the history of computations so we can retrace our steps later.

4. **Topological Sort:**
To calculate gradients properly, we can't just go in random order. We need a way to visit nodes such that we only calculate a parent's gradient after all its children are finished.

With this blueprint in hand, let's look at how to translate these concepts into clean, memory-safe C++.


### The Memory Management
Now that the fundamentals are clear, we face our first engineering challenge: Memory Management.

In Python, you can write `a = b + c`, and the language automatically keeps `b` and `c` alive as long as `a` needs them. In C++, stack variables die when they go out of scope. If we aren't careful, our computational graph will point to dead memory, leading to segmentation faults.

To solve this, we will use the Body/Handle pattern.

- **The Body [(ValueImpl)](#valueimpl-class)**: The heavy object that stores the data and graph connections. It lives on the Heap.

- **The Handle [(Value)](#value-class)**: A lightweight wrapper that you hold. It simply points to the Body and contains the methods the user will call.


### ValueImpl Class
Lets first start with the "Body" of our architecture. This is the private memory where the actual state lives.  

We define it as a simple struct rather than a class because it acts purely as a data container. It has no complex logic or invariant checks; it exists solely to hold the state that our public `Value` class will manage.

```c++
// struct inside value.cc
struct ValueImpl {
  // The payload
  float data;
  float grad;
  bool requires_grad;

  // The graph topology: Who created me?
  std::vector<std::shared_ptr<ValueImpl>> parents;

  // The chain rule logic: How do I pass gradients back?
  std::function<void()> backward_fn;

  // Constructor
  ValueImpl(float d, bool req_grad) 
  : data(d), grad(0.0f), requires_grad(req_grad) {}
};
```

This compact structure is the heart of the engine. Every instance of `ValueImpl` represents a single node in our computational graph. It holds: 
- The payload (the value of the number we are storing and its gradient), 
- The graph topology (pointers to the parents that created it), 
- The chain rule logic (a stored function that knows exactly how to calculate the local derivative). We will cover how this `backward_fn` works soon.

With just this structure, we have everything needed to perform backpropagation. The rest of the code is simply building the graph and traversing it.

### Value Class

Lets continue with the implementation of the actual wrapper:
```c++
// value.h
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

  // Operators (We will add more in the future)
  friend Value operator+(const Value& a, const Value& b);
  friend Value operator-(const Value& a, const Value& b);
  friend Value operator*(const Value& a, const Value& b);
  friend Value operator/(const Value& a, const Value& b);

 private:
  // Private constructor for internal graph building
  explicit Value(std::shared_ptr<ValueImpl> node);

  std::shared_ptr<ValueImpl> node_;
};
```

This class only contains one data member: a shared pointer to a `ValueImpl` struct. And contains some member functions that allow building our required ingredients:
- The operator overloads to hijack when a user types `Value a = b + c`, 
- The `backward()` function that will trigger the backpropagation algorithm
- The `zero_grad()` which resets the current gradients of the graph.


### Building the Graph

This is where the magic happens. When we add two Value objects, we don't just return the sum. We build a node that remembers its parents and defines a closure (lambda) for the backward pass.

```c++
Value operator+(const Value& a, const Value& b) {
  // 1. Compute the forward pass
  float out_data = a.data() + b.data();
  
  // 2. Create the new node
  bool req_grad = a.requires_grad() || b.requires_grad();
  auto out_node = std::make_shared<ValueImpl>(out_data, req_grad);

  if (req_grad) {
    // 3. Build the graph connections
    out_node->parents = {a.node_, b.node_};

    // 4. Define the Backward Step (Closure)
    // We capture the parents (node_a, node_b) so we can update their gradients later.
    auto node_a = a.node_;
    auto node_b = b.node_;
    std::weak_ptr<ValueImpl> weak_out = out_node;

    out_node->backward_fn = [weak_out, node_a, node_b]() {
      auto out = weak_out.lock();
      if (!out) return;

      // The Chain Rule for Addition:
      // f(x, y) = x + y
      // df/dx = 1 * global_grad
      // df/dy = 1 * global_grad
      if (node_a->requires_grad) node_a->grad += 1.0f * out->grad;
      if (node_b->requires_grad) node_b->grad += 1.0f * out->grad;
    };
  }
  
  return Value(out_node);
}
```

Notice the power of the Lambda. We are embedding the derivative logic directly into the graph node at the moment of creation. The node doesn't just know it was an addition; it possesses a function that knows exactly how to propagate gradients for addition.

This exactly what we wanted in our ingredients (2 and 3) to craft the engine, hijack the operators and build the DAG.

Also have in mind that this would be similarly performed when overloading other binary operations, it only requires changing the way the derivative is computed. For brievety I will not show the code of the rest of operators, you can check them in commit: Version 0.


### The backward pass

Once we have built the graph, backward() is surprisingly simple. We just need to traverse the nodes in reverse order (from output to input) and trigger those stored lambdas.

We use Topological Sort to ensure we don't process a parent before we have finished summing up all the gradients from its children. If you are un familiar with Topological Sort, you can read more [here](https://www.geeksforgeeks.org/dsa/topological-sorting/).

```c++
void Value::backward() {
  // 1. Seed the gradient (dL/dL = 1)
  node_->grad = 1.0f;

  // 2. Build the topological order
  std::vector<ValueImpl*> topo;
  std::unordered_set<ValueImpl*> visited;
  BuildTopo(node_.get(), topo, visited); // Standard DFS implementation

  // 3. Run the chain rule in reverse
  for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
    if ((*it)->backward_fn) {
      (*it)->backward_fn();
    }
  }
}
```

### Sanity check
Does it actually work? Let's try the example we derived by hand in the previous chapter: $f = (a + b) \cdot b$.
If $a=2, b=3$, we expect $\frac{\partial f}{\partial a} = 3$ and $\frac{\partial f}{\partial b} = 8$.

```c++
int DocsExample() {
  Value a(2.0f, true);
  Value b(3.0f, true);

  Value c = a + b; // c is 5
  Value f = c * b; // f is 15

  f.backward();

  std::cout << "f data: " << f.data() << "\n"; // Prints 15
  std::cout << "a grad: " << a.grad() << "\n"; // Prints 3
  std::cout << "b grad: " << b.grad() << "\n"; // Prints 8
}
```

It works as expected.

### The First Learning Loop: Fitting a Line
Sanity checks are useful, but they don't prove our engine can learn. To do that, we need to solve an optimization problem.  

We will attempt the "Hello World" of Machine Learning: Linear Regression.  

We will generate data that follows the pattern $y = 3x + 2$. We will then create a model with two parameters, $w$ (weight) and $b$ (bias), initialized to random values (or zero). Our goal is for the engine to automatically figure out that $w$ should be $3.0$ and $b$ should be $2.0$, solely by minimizing the error.

#### The Training Step
Since we are working with scalars, we can't do matrix multiplications yet. We have to manually loop through our data. The process for a single step of Gradient Descent is:
1. **Forward Pass:** Calculate the prediction $\hat{y} = wx + b$
2. **Calculate Loss:** Compute the Mean Squared Error (MSE) $= (\hat{y} - y)^2$
3. **Zero Gradients:** Clear old gradients from the previous step
4. **Backward Pass:** Call loss.backward() to populate gradients
5. **Optimizer Step:** Nudge $w$ and $b$ in the opposite direction of the gradient: $w = w - \eta \cdot \nabla w$

Where $y$ is the true label (ground truth), $x$ is the model's input, and $\eta$ is our learning rate (how much do we optimize in every step).

```c++
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
```


If you run this, you will see the loss plummet and the parameters $w$ and $b$ "magically" drift toward $3$ and $2$: 

<div align="center">

| Epoch | Loss (MSE)  | $w$      | $b$      |
|-------|-------------|----------|----------|
| 0     | 163.5       | 1.44     | 0.46     |
| 30    | 0.158108    | 3.15094  | 1.67772  |
| 60    | 0.0372749   | 3.07329  | 1.84352  |
| 90    | 0.00878777  | 3.03559  | 1.92402  |
| 120   | 0.00207176  | 3.01728  | 1.96311  |
| 150   | 0.000488446 | 3.00839  | 1.98209  |
| 180   | 0.000115148 | 3.00407  | 1.9913   |  

Final Results: $w = 3.00258, b = 1.9945$  

</div>


Congratulations! You have built the core mechanism used inside modern Deep Learning frameworks. It may only handle single numbers right now, but the logic; graph construction, backpropagation, and gradient descent is exactly the same core mechanism used inside systems like PyTorch, TensorFlow, JAX, and large language models. 

Now, we have a problem. Training a simple line took a loop. Training a neural network with a million weights iterating through every input parameter using for loops will be impossibly slow.

It is time to implement **Tensors**.

## Version 1: Tensors
In this chapter we will cover the weaknesess of our previous version, and how to solve some of them. 

We will also talk about tensors, good practices for cleaning our engine architecture and we will provide the same linear regression training loop as in the previous version, and you will see why this is still not enough.

### What in the world is a tensor?
This word may seem scary if you never heard it, but its simpler than it seems.
In physics, a tensor is a geometric object with complex definitions involving coordinate transformations. In Machine Learning, we use a much simpler definition:

- A Tensor is just a generalized n-dimensional array.
- A 0-D Tensor is a Scalar (just a number, like Value).
- A 1-D Tensor is a Vector (a list of numbers).
- A 2-D Tensor is a Matrix (a grid of numbers).
- A 3-D Tensor is a Cube of numbers, and so on.

By wrapping a std::vector<float> instead of a single float, we gain two massive advantages:

1. Memory Efficiency: We create ONE graph node (ComputeNode) that represents millions of numbers. The graph becomes 1,000,000x smaller.

2. Computational Efficiency: CPUs and GPUs love arrays. They have special instructions (SIMD) that can multiply 8, 16, or even more numbers simultaneously.


### The Architecture Problem (and the solution: Tensors)
If you look at our Version 0 implementation, you might notice something terrifying. To perform a simple calculation like $y = 3x + 2$, we are creating a graph node for every single number.

In main.cc, our weights w and b are single floating-point numbers. If we wanted to build a neural network with 1,000 parameters, we would need 1,000 Value objects. A modern Large Language Model has billions of parameters. If we tried to represent Llama-3 using our Value class, we would run out of RAM before we even finished loading the first layer, and the overhead of traversing billions of pointers would bring the CPU to a crawl.

We need to stop thinking in scalars (single numbers) and start thinking in Arrays.


### The Code Translation
How do we move from Version 0 to Version 1? We need to upgrade our internal storage.

In Version 0, our `ValueImpl` held a single float. In Version 1, we upgrade this to hold a `vector` and a `shape`.

#### The new Body: ComputeNode
We rename ValueImpl to ComputeNode to sound more professional (and accurate). It no longer holds a `float data`, but a `std::vector<float>`.

```c++
// compute_node.h (Version 1)
struct ComputeNode {
  std::vector<float> data;
  std::vector<float> grad;
  bool requires_grad;
  
  // Graph topology (Same as before)
  std::vector<std::shared_ptr<ComputeNode>> parents;
  std::function<void()> backward_fn;
  
  ComputeNode(const std::vector<float>& data_, bool requires_grad_) 
  : data(data_), grad(data_.size(), 0.0f), requires_grad(requires_grad_) {}
};
```

#### The new Handle: 

Our `Value` class becomes `Tensor`. It looks very similar, but now we track shapes.
```c++
// tensor.h
class Tensor {
 public:
  // Now we need to know the shape!
  Tensor(const std::vector<float>& data, std::vector<int> shape, bool requires_grad = false);

  // ... accessors ...

  // Operators now work on Arrays!
  friend Tensor operator+(const Tensor& a, const Tensor& b);
  friend Tensor matmul(const Tensor& a, const Tensor& b);
  
 private:
  // Design decision: shape is stored in Tensor, not in ComputeNode
  std::vector<int> shape_; // e.g., {2, 3} for a 2x3 matrix
  std::shared_ptr<ComputeNode> node_;
};
```

#### Operations on the Tensor Level
This is the toughest conceptual shift. In Version 0, `a + b` was just `float + float`. Now, `a + b` is an element-wise addition of two arrays.
We need to define both the Forward pass (how to calculate the result) and the Backward pass (how to propagate gradients) for arrays.

Let's look at Multiplication. In the scalar world, if $y = a \cdot b$, then $\frac{dy}{da} = b$.

In the Tensor world, if we perform element-wise multiplication, the logic is identical, just applied to every element in the vector. Here is the actual implementation in `ops.cc`:

```c++
// ops.cc

// --- MULTIPLICATION ---
// Forward: Simply multiply every element i
std::vector<float> MulForward(const std::vector<float>& a, const std::vector<float>& b) {
  assert(a.size() == b.size());
  std::vector<float> out(a.size());
  for (size_t i = 0; i < a.size(); ++i) out[i] = a[i] * b[i];
  return out;
}

// Backward: Propagate gradients element-wise
void MulBackward(Tensor& out, const Tensor& a, const Tensor& b) {
  const auto& grad_out = out.grad();
  
  if (a.requires_grad()) {
    auto& ga = const_cast<Tensor&>(a).mutable_grad();
    const auto& db = b.data();
    // dL/da = dL/dout * b
    for (size_t i = 0; i < ga.size(); ++i) ga[i] += grad_out[i] * db[i];
  }
  
  if (b.requires_grad()) {
    auto& gb = const_cast<Tensor&>(b).mutable_grad();
    const auto& da = a.data();
    // dL/db = dL/dout * a
    for (size_t i = 0; i < gb.size(); ++i) gb[i] += grad_out[i] * da[i];
  }
}
```

#### Matrix Multiplication (The Big Boss)
Element-wise stuff is easy. The real reason we use Tensors is Matrix Multiplication.If we have $Out = A \cdot B$ (Matrix A times Matrix B), how do we backpropagate the gradient? 
- $A$ is $(M \times K)$ 
- $B$ is $(K \times N)$
- $Out$ is $(M \times N)$  

We can't just do element-wise multiplication because every single output element $Out_{ij}$ is a sum of products involving the entire $i$-th row of $A$ and $j$-th column of $B$.
The derivatives are sums (dot products):
1. For A: $\frac{\partial L}{\partial A} = \frac{\partial L}{\partial Out} \cdot B^T$

2. For B: $\frac{\partial L}{\partial B} = A^T \cdot \frac{\partial L}{\partial Out}$  

Instead of implementing a Transpose operator and creating new nodes (which is slow), we can implement these loops directly. This gives us a raw look at how the gradients accumulate:
```c++
// ops.cc: Matrix Multiplication Backward
void MatMulBackward(Tensor& out, const Tensor& a, const Tensor& b) {
  const auto& grad_out = out.grad();  // shape: (M, N)
  const auto& shape_a = a.shape();
  const auto& shape_b = b.shape();
  
  int M = shape_a[0];
  int K = shape_a[1];
  int N = shape_b[1];
  
  // 1. Calculate Gradient for A (shape M x K)
  // dA[i, k] += sum_j( dOut[i, j] * B[k, j] )
  // This is effectively dOut * B_transpose
  if (a.requires_grad()) {
    auto& ga = const_cast<Tensor&>(a).mutable_grad();
    const auto& b_data = b.data();
    
    for (int i = 0; i < M; ++i) {
      for (int k = 0; k < K; ++k) {
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
          sum += grad_out[i * N + j] * b_data[k * N + j];
        }
        ga[i * K + k] += sum;
      }
    }
  }
  
  // 2. Calculate Gradient for B (shape K x N)
  // dB[k, j] += sum_i( A[i, k] * dOut[i, j] )
  // This is effectively A_transpose * dOut
  if (b.requires_grad()) {
    auto& gb = const_cast<Tensor&>(b).mutable_grad();
    const auto& a_data = a.data();
    
    for (int k = 0; k < K; ++k) {
      for (int j = 0; j < N; ++j) {
        float sum = 0.0f;
        for (int i = 0; i < M; ++i) {
          sum += a_data[i * K + k] * grad_out[i * N + j];
        }
        gb[k * N + j] += sum;
      }
    }
  }
}
```
This code is the "heart" of the engine. It manually routes the error signals ($\text{grad\_out}$) back to the inputs (a and b) by respecting the linear algebra rules.



#### Clean Architecture: Ops and Compute
In Version 0, we stuffed the logic for addition directly inside the `operator+` function. This gets messy fast. What if we want to implement Matrix Multiplication, ReLU, Sigmoid, Softmax, Log, and Power? Our `tensor.cc` file would be thousands of lines long.

In Version 1, we decouple the Data from the Operations.
1. `compute_node.h`: Only holds state. It doesn't know what "addition" is.
2. `ops.cc`: Contains the actual math.

We define a new member function in tensor to build the graph nodes and apply any operation
```c++
Tensor Tensor::ApplyBinaryOp(const Tensor& a, const Tensor& b,
                             const std::string& op_name,
                             BinaryMathOp forward_op,
                             BinaryGradOp backward_op) {
  std::vector<float> out_data = forward_op(a.data(), b.data());
  std::vector<int> out_shape = a.shape(); // Assuming same shape for now

  bool req_grad = a.requires_grad() || b.requires_grad();
  auto out_node = std::make_shared<ComputeNode>(out_data, req_grad);
  out_node->op_name = op_name;

  if (req_grad) {
    out_node->parents.push_back(a.node_);
    out_node->parents.push_back(b.node_);

    auto node_a = a.node_;
    auto node_b = b.node_;
    
    auto shape_a = a.shape_; 
    auto shape_b = b.shape_;
    
    // Use weak_ptr to break the cycle
    std::weak_ptr<ComputeNode> weak_out = out_node;

    out_node->backward_fn = [weak_out, node_a, node_b, backward_op, out_shape, shape_a, shape_b]() {
      auto out_ptr = weak_out.lock(); 
      if (!out_ptr) return;

      Tensor t_out(out_shape, out_ptr);
      Tensor t_a(shape_a, node_a);
      Tensor t_b(shape_b, node_b);
      backward_op(t_out, t_a, t_b);
    };
  }
  return Tensor(out_shape, out_node);
}
```

To call every operation, just use the generic wrapper that uses the forward and backward function of every operation.
```c++
Tensor operator+(const Tensor& a, const Tensor& b) {
  return Tensor::ApplyBinaryOp(a, b, "Add", AddForward, AddBackward);
}

Tensor operator-(const Tensor& a, const Tensor& b) {
  return Tensor::ApplyBinaryOp(a, b, "Sub", SubForward, SubBackward);
}

Tensor operator-(const Tensor& a) { // Unary operator
  return Tensor::ApplyUnaryOp(a, "Neg", NegForward, NegBackward);
}
// ... and so on ...
```


For example, look at how clean the Addition logic is when isolated in `ops.cc`:

```c++
// ops.cc

// The Forward Pass (Just Math)
std::vector<float> AddForward(const std::vector<float>& a, const std::vector<float>& b) {
  // ... assertions ...
  std::vector<float> out(a.size());
  for (size_t i = 0; i < a.size(); ++i) out[i] = a[i] + b[i];
  return out;
}

// The Backward Pass (Just Gradients)
void AddBackward(Tensor& out, const Tensor& a, const Tensor& b) {
  const auto& grad_out = out.grad();
  if (a.requires_grad()) {
    // Add gradient to A
  }
  if (b.requires_grad()) {
    // Add gradient to B
  }
}

```

The Tensor class essentially just calls these functions. This separation of concerns allows us to easily plug in new operations without breaking the core graph engine.

#### Main Concerns: Why does this still look ugly?
We have Tensors! We can do matrix multiplication! So our main training loop should be beautiful, right?

Well... look at `main.cc`.

```c++
  // main.cc
  Tensor weights({0.0f, 0.0f}, {2, 1}, true); // Weights are a 2x1 Column Vector

  for (int epoch = 0; epoch < 2000; ++epoch) { 
    float total_loss = 0.0f;

    // THE UGLY PART: We are still looping!
    for (size_t i = 0; i < X_data.size(); ++i) {
      Tensor x(X_data[i], {1, 2}, false); 
      Tensor y({Y_data[i]}, {1, 1}, false); 

      // Prediction: (1x2) * (2x1) = (1x1)
      Tensor prediction = matmul(x, weights); 
      
      // ... calculate loss ...
    }
    // ...
  }
```
Wait, why are we looping through X_data one row at a time?

Ideally, we want to load the entire dataset into a single $(N \times 2)$ matrix and multiply it by our weights in one go: 
$$Y_{pred} = X_{batch} \cdot W$$
If we tried to do that right now, our code would crash. Why?
#### Shape Mismatch
If we multiply $(N \times 2)$ by $(2 \times 1)$, we get an $(N \times 1)$ output. But if we try to subtract our target $Y$ (which might be a different shape or format), our current simple SubForward function will assert that the shapes must be identical.
Because we lack **Broadcasting**, we are forced to treat every sample individually, effectively crippling the performance gains of our Tensors.

But don't worry. Solving this is exactly what we will do in the next chapter.

## Version 2: Broadcasting

In the last chapter, we ended on a cliffhanger. We had Tensors, we had `matmul`, and yet our training loop was still ugly because of a sneaky little requirement buried inside every one of our forward functions: `assert(a.size() == b.size())`.

This chapter is about getting rid of that assertion, safely.

### What is Broadcasting, anyway?

Let's say you have a batch of 32 samples, each with 4 features, stored as a $(32, 4)$ tensor. You also have a bias vector of shape $(4,)$ — one bias value per feature. Conceptually, you want to add the bias to *every row* of the batch:

$$Out_{ij} = X_{ij} + bias_j$$

Mathematically this is obvious. But our `AddForward` from Version 1 has no idea what to do with two vectors of different lengths ($32 \times 4 = 128$ elements vs. $4$ elements). It would either crash on the assertion, or worse, silently read out of bounds.

The naive fix would be to physically copy the bias 32 times so it becomes a $(32, 4)$ tensor, and then add normally. That works, but it wastes memory and time — and you'd have to remember to do this *everywhere*: addition, subtraction, multiplication, division...

**Broadcasting** is the trick that NumPy, PyTorch, and friends use to avoid those copies. The rule is simple:

> Two dimensions are compatible if they are equal, or if one of them is 1. When one of them is 1, that dimension is "stretched" (virtually, not physically) to match the other.

So $(32, 4)$ and $(4,)$ are compatible: align them from the right, treat the missing leading dimension of the second tensor as a $1$, giving $(1, 4)$, and since $1$ can stretch to match $32$, the broadcast output shape is $(32, 4)$.

This is exactly the rule we need to implement in three pieces:
1. **Figure out the output shape** given two input shapes.
2. **Map an index in the (large) output** back to the corresponding index in a (smaller, broadcasted) input.
3. **Collapse a gradient** computed at the output's shape back down to a smaller input's shape during the backward pass.

Let's tackle them one at a time.

### `ComputeBroadcastShape`: Aligning shapes from the right

The first helper answers the question: "If I combine a tensor of shape `shape_a` with one of shape `shape_b`, what shape does the result have?"

```c++
// core/ops.cc (Version 2)

// 1. Helper: Compute the result shape (e.g., [32, 10] + [1, 10] -> [32, 10])
std::vector<int> ComputeBroadcastShape(const std::vector<int>& shape_a, 
                                       const std::vector<int>& shape_b) {
  std::vector<int> out_shape;
  auto it_a = shape_a.rbegin(); 
  auto it_b = shape_b.rbegin();
  
  while (it_a != shape_a.rend() || it_b != shape_b.rend()) {
    int dim_a = (it_a != shape_a.rend()) ? *it_a : 1; // Treat missing dims as 1
    int dim_b = (it_b != shape_b.rend()) ? *it_b : 1;
    
    if (dim_a == dim_b) {
      out_shape.insert(out_shape.begin(), dim_a);
    } else if (dim_a == 1) {
      out_shape.insert(out_shape.begin(), dim_b);
    } else if (dim_b == 1) {
      out_shape.insert(out_shape.begin(), dim_a);
    } else {
      throw std::runtime_error("Incompatible dimensions for broadcasting");
    }
    
    if (it_a != shape_a.rend()) ++it_a;
    if (it_b != shape_b.rend()) ++it_b;
  }
  return out_shape;
}
```

The implementation walks both shapes **backwards** (`rbegin()`/`rend()`) — this is the "align from the right" rule. If one shape runs out of dimensions before the other, we just pretend it has a `1` there. For each pair of dimensions, we pick whichever one is bigger, unless they're incompatible (neither equal nor 1), in which case we throw — better to fail loudly at graph-construction time than silently corrupt memory later.

Example: `shape_a = {32, 4}`, `shape_b = {4}`.
- Rightmost: `4` vs `4` → equal → `4`.
- Next: `32` vs *(missing, treated as `1`)* → `32`.
- Result: `{32, 4}`. 

### `MapIndex`: From a flat output index to an input index

Now for the trickier part. Our tensors are stored as flat `std::vector<float>`, using row-major order. If the *output* has shape $(32, 4)$ and is 128 elements long, but our *bias* tensor only has 4 elements, how do we know which of the 4 bias values corresponds to output element #57?

That's the job of `MapIndex`:

```c++
namespace {

// TODO: Optimize
// 2. Helper: Map output index to input index. Note: this implementation is not
// efficient, but done with educational purposes
size_t MapIndex(size_t flat_index, const std::vector<int>& out_shape, const std::vector<int>& in_shape) {
  size_t in_index = 0;
  size_t in_stride = 1;
  
  // Align dimensions to the right
  int dim_offset = out_shape.size() - in_shape.size();

  for (int d = out_shape.size() - 1; d >= 0; --d) {
    int out_dim_size = out_shape[d];
    int coord = flat_index % out_dim_size; // Coordinate in this dimension
    flat_index /= out_dim_size; 

    int in_d = d - dim_offset;
    if (in_d >= 0) {
      int in_dim_size = in_shape[in_d];
      // If broadcast (size 1), coord is 0. Else, matches output.
      int in_coord = (in_dim_size == 1) ? 0 : coord;
      in_index += in_coord * in_stride;
      in_stride *= in_dim_size;
    }
  }
  return in_index;
}

}
```

Let's unpack this with the same example: `out_shape = {32, 4}`, `in_shape = {4}` (our bias), and say `flat_index = 57`.

The function walks the output dimensions from **last to first** (right to left, same direction as before), peeling off one coordinate at a time using `%` and `/` — the classic trick for converting a flat index into multi-dimensional coordinates:

- $d=1$ (size 4): `coord = 57 % 4 = 1`, `flat_index` becomes `57 / 4 = 14`. This dimension exists in `in_shape` too (`in_d = 0`), and its size there is also `4` (not a broadcast dim), so `in_coord = coord = 1`. `in_index = 1`, `in_stride = 4`.
- $d=0$ (size 32): `coord = 14 % 32 = 14`. But `in_d = -1` — the bias tensor has no dimension here (it was "padded" with an implicit `1`). We skip it entirely; it doesn't contribute to `in_index`.

Final result: `in_index = 1`. So output element 57 (which lives in row 14, column 1 of the $32\times4$ grid) reads from `bias[1]` — exactly what we'd expect, since every row reuses the same 4 bias values.

The key insight is the line `int in_coord = (in_dim_size == 1) ? 0 : coord;`. Whenever the input has a size-1 dimension (a broadcast dimension), we always read index `0` along that axis — that's the "virtual stretching". Combined with `dim_offset`, dimensions that don't exist at all in the smaller tensor are simply ignored.

With `ComputeBroadcastShape` and `MapIndex`, our forward passes barely change. Here's the new `AddForward`:

```c++
// core/ops.cc (Version 2)
std::vector<float> AddForward(const std::vector<float>& a, 
                              const std::vector<float>& b, 
                              const std::vector<int>& shape_a,
                              const std::vector<int>& shape_b) {
  std::vector<int> out_shape = ComputeBroadcastShape(shape_a, shape_b);
  size_t total_size = std::accumulate(out_shape.begin(), out_shape.end(), 1, std::multiplies<int>());

  std::vector<float> out(total_size);
  for (size_t i = 0; i < total_size; ++i) {
    size_t idx_a = MapIndex(i, out_shape, shape_a);
    size_t idx_b = MapIndex(i, out_shape, shape_b);
    out[i] = a[idx_a] + b[idx_b];
  }
  return out;
}
```

Notice the forward functions now take the *shapes* as extra arguments — they need them to compute `out_shape` and to map indices. Every binary forward op (`Add`, `Sub`, `Mul`, `Div`) follows this same template now.

### `SumToShape`: Reducing gradients back down

Forward pass is only half the story. What about backward?

Think about our bias example again: `bias` has shape $(4,)$, but the output (and therefore `grad_out`) has shape $(32, 4)$. During backpropagation, *every one* of the 32 rows contributed to the use of `bias[j]`, so the gradient of `bias[j]` must be the **sum** of all 32 contributions:

$$\frac{\partial L}{\partial bias_j} = \sum_{i=0}^{31} \frac{\partial L}{\partial Out_{ij}}$$

This is the reverse operation of broadcasting: broadcasting *replicates* a value across many output positions during the forward pass, so backpropagation must *accumulate* (sum) gradients from all those positions back into the single original value.

```c++
// core/ops.cc (Version 2)

// Sums a large gradient down to a smaller shape (for broadcasting backward pass)
std::vector<float> SumToShape(const std::vector<float>& grad, 
                              const std::vector<int>& large_shape, 
                              const std::vector<int>& small_shape) {
  // 1. Calculate size of the small tensor
  size_t small_size = std::accumulate(small_shape.begin(), small_shape.end(), 1, std::multiplies<int>());
  
  // 2. Prepare output gradient (init to 0.0)
  std::vector<float> reduced_grad(small_size, 0.0f);
  
  // 3. Iterate over the LARGE gradient
  for (size_t i = 0; i < grad.size(); ++i) {
    // Determine which index in the small tensor this gradient element belongs to
    size_t idx = MapIndex(i, large_shape, small_shape);
    
    // Accumulate!
    reduced_grad[idx] += grad[i];
  }
  
  return reduced_grad;
}
```

It reuses `MapIndex` to figure out, for every element of the *large* gradient, which element of the *small* tensor it should be accumulated into. Beautifully, the exact same mapping logic that we used to "spread" values during forward is reused to "collect" gradients during backward — just with the roles of source and destination swapped.

`AddBackward` now looks like this:

```c++
// core/ops.cc (Version 2)
void AddBackward(Tensor& out, const Tensor& a, const Tensor& b) {
  const std::vector<float>& grad_out = out.grad();
  std::vector<int> shape_out = out.shape();

  if (a.requires_grad()) {
    std::vector<float> grad_a_update;
    if (a.shape() == shape_out) grad_a_update = grad_out;
    else grad_a_update = SumToShape(grad_out, shape_out, a.shape());

    auto& ga = const_cast<Tensor&>(a).mutable_grad();
    for (size_t i = 0; i < ga.size(); ++i) ga[i] += grad_a_update[i];
  }

  if (b.requires_grad()) {
    std::vector<float> grad_b_update;
    if (b.shape() == shape_out) grad_b_update = grad_out;
    else grad_b_update = SumToShape(grad_out, shape_out, b.shape());

    auto& gb = const_cast<Tensor&>(b).mutable_grad();
    for (size_t i = 0; i < gb.size(); ++i) gb[i] += grad_b_update[i];
  }
}
```

The `if (a.shape() == shape_out)` check is just a fast path: if a tensor's shape already matches the output (no broadcasting happened for that operand), we skip the reduction entirely and copy the gradient directly.

#### A subtlety for multiplication and division

For `+` and `-`, broadcasting only affects *which* gradient values go where — the local derivative is always $1$ (or $-1$). But for `*` and `/`, the local derivative depends on the *other* operand's value, which might itself need broadcasting!

Look at `MulBackward`:

```c++
// core/ops.cc (Version 2)
void MulBackward(Tensor& out, const Tensor& a, const Tensor& b) {
  const auto& grad_out = out.grad();
  std::vector<int> shape_out = out.shape();
  if (a.requires_grad()) {
    // 1. Compute the expanded gradient (dL/dY * B)
    std::vector<float> temp_grad(grad_out.size());
    const auto& b_data = b.data();
    for (size_t i = 0; i < grad_out.size(); ++i) {
      size_t idx_b = MapIndex(i, shape_out, b.shape()); // Virtually expand B
      temp_grad[i] = grad_out[i] * b_data[idx_b];
    }

    // 2. Reduce (Sum) if A is smaller than output
    std::vector<float> final_grad_a;
    if (a.shape() == shape_out) final_grad_a = temp_grad;
    else final_grad_a = SumToShape(temp_grad, shape_out, a.shape());

    // 3. Sum gradients to A
    auto& ga = const_cast<Tensor&>(a).mutable_grad();
    for (size_t i = 0; i < ga.size(); ++i) ga[i] += final_grad_a[i];
  }
  // ... symmetric logic for b ...
}
```

It's a two-step dance:
1. **Expand**: First compute the gradient *at the output's shape*, using `MapIndex` to "virtually broadcast" `b`'s values up to the output size (`b_data[idx_b]`).
2. **Reduce**: Then call `SumToShape` to collapse that expanded gradient back down to `a`'s (possibly smaller) shape.

This two-step "expand, then reduce" pattern shows up in `DivBackward` too — division's local derivatives ($\partial(a/b)/\partial a = 1/b$ and $\partial(a/b)/\partial b = -a/b^2$) depend on the other operand, so the same dance is needed.

### Sanity check

With broadcasting in place, our XOR network in `main.cc` can finally do this without crashing:

```c++
// main.cc
Tensor h_pre = matmul(X, W1) + b1; // X: {4,2} @ W1: {2,4} = {4,4}, b1: {4} -> broadcasts to {4,4}!
```

`matmul(X, W1)` produces a $(4, 4)$ tensor (4 samples, 4 hidden units), and `b1` is just a $(4,)$ vector — one bias per hidden unit, shared across all 4 samples. `ComputeBroadcastShape({4,4}, {4})` gives `{4, 4}`, `MapIndex` correctly repeats `b1` for every row, and `SumToShape` correctly accumulates `b1`'s gradient across all 4 samples during backward.

We can finally process an entire batch in one shot, with no Python-style for-loops over samples. The engine is starting to look like a *real* framework.

But there's still a problem lurking: every time we call `ApplyBinaryOp` or `ApplyUnaryOp`, we allocate a brand new `ComputeNode` on the heap with `std::make_shared`. In a training loop running thousands of epochs, each performing dozens of operations, that's a *lot* of heap churn. Let's fix that next.

## Version 3: From a Toy to a Tool

By the end of Version 2, mini-autodiff could do everything *conceptually* required of a deep learning engine: build a graph, broadcast shapes, backpropagate gradients through a batch. But there's a wide gulf between "works" and "is a tool you'd actually want to use". Version 3 closes that gulf along four fronts: performance, ergonomics (optimizers & losses), project hygiene (structure & tests), and correctness (edge cases & CI).

Let's go through them in the order that makes the most sense pedagogically — starting with what's happening under the hood every time we run a forward pass.

### Memory: Stop hammering the heap

#### The problem

Every single arithmetic operation — every `+`, every `relu`, every `matmul` — creates a new `ComputeNode`. In Version 2, that meant a call to `std::make_shared<ComputeNode>(...)`, which under the hood performs a heap allocation.

Now picture our XOR training loop: 1000 epochs, each doing ~6 tensor operations (two matmuls, two adds, a relu, a loss). That's 6000 heap allocations and 6000 deallocations *per run* — just for a tiny 4-sample dataset. Heap allocation isn't free; it involves locking, bookkeeping, and potential fragmentation. For a real network with millions of parameters and thousands of steps, this overhead adds up fast.

The classic fix for "I keep allocating and freeing objects of the exact same size, over and over" is an **object pool**.

#### `ObjectPool` and `PoolAllocator`

```c++
// core/memory_pool.h (Version 3)

// A simple thread-safe object pool
template <typename T>
class ObjectPool {
 public:
  static ObjectPool& Instance() {
    static ObjectPool instance;
    return instance;
  }

  T* Allocate() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!pool_.empty()) {
      T* ptr = pool_.back();
      pool_.pop_back();
      return ptr;
    }
    // Allocate raw memory
    return static_cast<T*>(::operator new(sizeof(T)));
  }

  void Deallocate(T* ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    pool_.push_back(ptr);
  }

  // Clear pool to free memory to OS
  void Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (T* ptr : pool_) {
      ::operator delete(ptr);
    }
    pool_.clear();
  }

 private:
  ObjectPool() = default;
  ~ObjectPool() { Clear(); }
  
  std::vector<T*> pool_;
  std::mutex mutex_;
};
```

The idea is almost embarrassingly simple: instead of giving memory back to the OS when a `ComputeNode` dies, we keep the raw block around in a `pool_` vector. The next time we need a `ComputeNode`-sized chunk of memory, we just hand back one we already have, skipping the call to `::operator new` entirely. It's a free-list — a classic technique, but it makes a real difference when the same-sized object is created and destroyed thousands of times per second.

To plug this into `std::shared_ptr` (which we're already using for `ComputeNode`), we wrap it in an STL-compatible allocator:

```c++
// core/memory_pool.h (Version 3)

// STL-compatible allocator for std::allocate_shared
template <typename T>
struct PoolAllocator {
  using value_type = T;

  PoolAllocator() = default;
  template <class U> constexpr PoolAllocator(const PoolAllocator<U>&) noexcept {}

  T* allocate(std::size_t n) {
    if (n > 1) return std::allocator<T>().allocate(n);
    return ObjectPool<T>::Instance().Allocate();
  }

  void deallocate(T* p, std::size_t n) {
    if (n > 1) {
      std::allocator<T>().deallocate(p, n);
    } else {
      ObjectPool<T>::Instance().Deallocate(p);
    }
  }
};
```

Why the `n > 1` check? `std::allocate_shared` *usually* asks the allocator for exactly one `T` (it bundles the control block and the object into a single allocation when `T`'s allocator is used directly). If for some reason a request for more than one object comes through, we fall back to the regular `std::allocator`. The pool is specifically tuned for the "one `ComputeNode` at a time" case, which is the overwhelmingly common one.

With this allocator, every place that used to say:

```c++
auto out_node = std::make_shared<ComputeNode>(out_data, req_grad);
```

now says:

```c++
// core/ops.cc (Version 3)
auto out_node = std::allocate_shared<ComputeNode>(
  PoolAllocator<ComputeNode>(),
  std::move(out_data), 
  std::move(out_shape), 
  req_grad
);
```

Two changes worth noticing: we swapped `make_shared` for `allocate_shared` with our pool, *and* we switched to `std::move` for the data and shape vectors. `out_data` and `out_shape` are temporaries we just computed — there's no reason to copy their contents into the node when we can simply transfer ownership.

#### Shrinking the lambdas

There's a second, more subtle memory win in this commit. Recall from Version 1 that every operation's `backward_fn` is a `std::function` — a closure capturing whatever it needs to recompute gradients later. In Version 1/2, `ApplyBinaryOp` captured the **shapes** of the inputs and output by value:

```c++
// Version 2 (before)
out_node->backward_fn = [weak_out, node_a, node_b, backward_op, out_shape, shape_a, shape_b]() {
  ...
  Tensor t_out(out_shape, out_ptr);
  Tensor t_a(shape_a, node_a);
  Tensor t_b(shape_b, node_b);
  backward_op(t_out, t_a, t_b);
};
```

Each `std::vector<int>` shape capture is itself a small heap allocation (or at least extra bytes inside the `std::function`'s internal storage, possibly forcing it to heap-allocate if it doesn't fit "in-place"). Multiply that by three shapes, times thousands of nodes, and it's wasted space and wasted allocations.

But wait — do we actually need to capture the shapes separately? By Version 2, `ComputeNode` itself stores its `shape` (see `compute_node.h`). So the shape is already reachable through `node_a`, `node_b`, and `out_ptr` — no need to duplicate it in the closure!

```c++
// core/tensor.cc (Version 3) — ApplyBinaryOp
auto node_a = a.node_;
auto node_b = b.node_;

std::weak_ptr<ComputeNode> weak_out = out_node;

// Optimization: Lambda Capture Size
// We NO LONGER capture 'out_shape', 'shape_a', or 'shape_b' by value.
// They are accessed via the node pointers.
out_node->backward_fn = [weak_out, node_a, node_b, backward_op]() {
  auto out_ptr = weak_out.lock(); 
  if (!out_ptr) return;

  // Reconstruct Tensors wrappers from raw nodes (cheap, no copies)
  Tensor t_out(out_ptr);
  Tensor t_a(node_a);
  Tensor t_b(node_b);
  
  backward_op(t_out, t_a, t_b);
};
```

`Tensor(node_a)` is now just a cheap wrapper around a `shared_ptr` we already have — the shape lives inside the node and is read through `t_a.shape()`. The closure shrinks from "two shared_ptrs + a weak_ptr + a function pointer + three vectors" down to "two shared_ptrs + a weak_ptr + a function pointer". Small per-node, but it adds up across an entire graph.

### Optimizers and Losses: Teaching the engine to learn on its own

So far, our "optimizer" was two lines of hand-written gradient descent at the bottom of the training loop:

```c++
w.mutable_data() -= learning_rate * w.grad();
b.mutable_data() -= learning_rate * b.grad();
```

That's fine for two parameters. It is not fine for a neural network with dozens of weight matrices and bias vectors. Version 3 introduces a proper `Optimizer` abstraction.

#### The `Optimizer` base class

```c++
// nn/optimizer.h (Version 3)
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
```

An optimizer just bundles together "the list of tensors I'm allowed to update" and "how fast I update them" (the learning rate `lr_`). It exposes two operations:

- **`zero_grad()`**: Before every backward pass, gradients must be reset to zero — remember, `backward()` *accumulates* into `grad`, it doesn't overwrite. Forgetting this is one of the most common bugs in any autodiff-based training loop (in Version 0 we had to do this by hand, for *every* parameter, *every* epoch).
- **`step()`**: The actual update rule, applied *after* `backward()` has populated `.grad()` on every parameter. This is left `virtual` because different optimizers update parameters differently.

#### SGD with momentum

The simplest concrete optimizer is Stochastic Gradient Descent. Plain SGD is just $w \leftarrow w - \eta \nabla w$ — exactly what we did by hand before. But Version 3's `SGD` also supports **momentum**, which helps the optimizer "build up speed" in directions where the gradient is consistently pointing, and dampen oscillations:

```c++
// nn/optimizer.cc (Version 3)
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
```

Think of `vel` (velocity) as a rolling average of past gradients. If the gradient keeps pointing the same direction, `vel` grows and the parameter moves further each step — like a ball rolling downhill picking up speed. If the gradient flips direction, momentum partially cancels out the previous velocity, smoothing the trajectory.

#### Adam: the modern default

SGD is simple but can be slow to converge, and picking a good learning rate is fiddly — too high and training diverges, too low and it crawls. **Adam** (Adaptive Moment Estimation) is the optimizer most modern training scripts reach for by default, because it adapts its effective step size *per parameter*, automatically.

```c++
// nn/optimizer.cc (Version 3)
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
```

You don't need to memorize this formula, but here's the intuition:

- **`m`** is an exponential moving average of the gradient itself — similar to SGD's momentum, it tracks "which direction has this parameter been moving recently?"
- **`v`** is an exponential moving average of the *squared* gradient — it tracks "how large/volatile have the gradients for this parameter been?"
- The update divides by $\sqrt{\hat{v}} + \epsilon$. If a parameter has had consistently huge gradients, $\hat{v}$ is large, and the effective step shrinks. If gradients have been tiny, the step grows. Each parameter effectively gets its own adaptive learning rate.
- The `_corr` terms exist because `m` and `v` start at zero and are biased toward zero in early steps — the correction compensates for that "cold start".

`eps_` (default `1e-8`) is just there to avoid dividing by zero when `v_hat` is zero.

#### Losses: MSE vs. BCE

A loss function turns "how wrong was my prediction" into a single scalar number that `backward()` can differentiate. Version 3 adds an abstract `Loss` interface plus two concrete losses:

```c++
// nn/loss.h (Version 3)

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
```

**MSE (Mean Squared Error)** is $\frac{1}{N}\sum (pred - target)^2$ — the same loss we used by hand back in Version 0's linear regression. It's the natural choice for **regression** problems, where the target is a continuous number (predict a price, a temperature, a coordinate...). Notice how concisely it reads now: `pred - target` uses our broadcasting-aware `operator-`, `diff * diff` squares element-wise, and `mean` reduces to a scalar — three operations we already built, composed into a loss.

**BCE (Binary Cross-Entropy)** is for **classification** problems where the target is `0` or `1` (e.g., "is this a cat?"), and `pred` is a probability produced by a `sigmoid`:

```c++
// nn/loss.h (Version 3)
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
```

The formula is:

$$L = -\frac{1}{N}\sum \left[ y \log(\hat{y}) + (1-y)\log(1-\hat{y}) \right]$$

For each sample, only one of the two terms is "active" (since $y$ is 0 or 1), and it penalizes confident-but-wrong predictions *much* more harshly than slightly-off ones — that's what makes it suited to classification. We'll come back to the `clamp` call in a moment; it's actually load-bearing, not decorative.

### Project structure: Growing up

By this point, mini-autodiff had a flat pile of `.cc`/`.h` files in the repo root. That's fine for a weekend prototype, but it doesn't scale, and it doesn't *teach* the right lesson about how real frameworks are organized. So Version 3 reorganizes everything into two directories:

```
core/   - the autodiff engine itself: tensor, compute graph, memory, ops
nn/     - higher-level API built on top of core: optimizers, losses
```

This mirrors a pattern you'll find in PyTorch, TensorFlow, and JAX: there's a **runtime/engine** layer that knows nothing about "neural networks" — just tensors, shapes, and gradients — and an **API** layer built on top that knows about layers, losses, and optimizers. `core/` doesn't (and shouldn't) `#include` anything from `nn/`; the dependency only goes one way. If you ever wanted to add a second "API" on top (say, a `Module`/`Layer` system), it would live next to `nn/`, both built on the same `core/`.

Build artifacts also got tidied up — `.o` and `.d` files now land in a `build/` directory instead of cluttering the source tree, mirroring the `core/`/`nn/` split:

```makefile
# Makefile (Version 3)
SRCS = main.cc core/tensor.cc core/ops.cc nn/optimizer.cc
OBJS = $(addprefix $(BUILD_DIR)/,$(SRCS:.cc=.o))
```

#### Gradient checking: trusting your derivatives

Here's an uncomfortable truth: it is *very* easy to write a `*Backward` function that compiles, runs, and produces a plausible-looking number — and is subtly wrong. A sign flip, a missing broadcast reduction, an off-by-one in `MapIndex`... any of these can silently make your network train *worse* than it should, without ever crashing.

The standard cure is **gradient checking**: compute the gradient two different ways and compare.

1. **Analytic gradient**: call `backward()` and read `.grad()` — this is what our engine computes via the chain rule.
2. **Numerical gradient**: nudge a single input element by a tiny amount $\epsilon$ in each direction and measure how much the output changes — the definition of a derivative, approximated via *central differences*:

$$\frac{\partial f}{\partial x_i} \approx \frac{f(x_i + \epsilon) - f(x_i - \epsilon)}{2\epsilon}$$

If these two numbers disagree by more than a small tolerance, something in our backward implementation is wrong. Version 3 adds exactly this as `tests/test_gradcheck.cc`:

```c++
// tests/test_gradcheck.cc (Version 3)

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
```

`GradCheck` is generic: give it a name, a function `f` that builds some `Tensor` expression out of `inputs`, and the `inputs` themselves. It runs `f`, calls `backward()`, then for every element of every input that `requires_grad()`, it perturbs that single number, re-runs `f` (forward only, no graph needed since we just read `.data()[0]`), and compares the finite-difference slope to what `backward()` computed.

This single harness is then reused to test every op we've built — `Add`, `Sub`, `Mul`, `Div` (same-shape *and* broadcast variants!), `matmul`, `relu`, `sigmoid`, `log`, `clamp`, `sum`, `mean` — plus `MSELoss` and `BCELoss`. Running it is as simple as:

```bash
make test
```

which builds `tests/test_gradcheck.cc` against everything in `core/` and `nn/` (but not `main.cc`), and runs the resulting binary.

This commit also tightened up `matmul`'s safety: `ValidateMatMulShapes` now explicitly checks that both operands are 2D and that their inner dimensions agree *before* `MatMulForward` starts indexing into raw vectors:

```c++
// core/ops.cc (Version 3)
void ValidateMatMulShapes(const std::vector<int>& shape_a, const std::vector<int>& shape_b) {
  if (shape_a.size() != 2 || shape_b.size() != 2) {
    throw std::invalid_argument("matmul: both tensors must be 2D");
  }
  if (shape_a[1] != shape_b[0]) {
    throw std::invalid_argument("matmul: inner dimensions must match (A is [M,K], B must be [K,N])");
  }
}
```

Before this check existed, passing a $(3,)$ vector and a $(3,)$ vector to `matmul` (neither is 2D!) would have caused `MatMulForward` to read `shape_a[1]`, which doesn't exist — undefined behavior. Now it's a clean, catchable `std::invalid_argument`.

### Correctness: The little bugs that bite hardest

Writing `tests/test_gradcheck.cc` immediately started paying for itself — it surfaced real bugs.

#### `clamp`: keeping `log` away from zero

Remember `BCELoss`'s comment about clamping? Here's why it matters. `BCELoss` computes `log(pred)` and `log(1 - pred)`. If a sigmoid output saturates to exactly `0.0` or `1.0` (which `float32` does for sufficiently large/small inputs), then `log(0.0)` is $-\infty$, and `0 * (-\infty)` is `NaN`. One `NaN` in a gradient is enough to poison an entire training run — every parameter it touches becomes `NaN` forever.

The fix is the new `clamp` op:

```c++
// core/ops.cc (Version 3)
// --- CLAMP ---
std::vector<float> ClampForward(const std::vector<float>& a, float min_val, float max_val) {
  std::vector<float> out(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    out[i] = std::min(std::max(a[i], min_val), max_val);
  }
  return out;
}

void ClampBackward(Tensor& out, const Tensor& a, float min_val, float max_val) {
  const auto& grad_out = out.grad();
  if (a.requires_grad()) {
    auto& ga = const_cast<Tensor&>(a).mutable_grad();
    const auto& da = a.data();
    for (size_t i = 0; i < ga.size(); ++i) {
      // Gradient is 1 inside the clamp range, 0 outside (flat regions)
      float derivative = (da[i] >= min_val && da[i] <= max_val) ? 1.0f : 0.0f;
      ga[i] += grad_out[i] * derivative;
    }
  }
}
```

`clamp(x, min, max)` just squashes values into `[min, max]`. Its gradient is `1` where the input was inside the range (the operation was a no-op there) and `0` where it clipped a value (the output no longer depends on the exact input — nudging it further wouldn't change anything). `BCELoss` uses this to keep predictions in `[1e-7, 1 - 1e-7]` — close enough to `0`/`1` to not affect the loss value meaningfully, but far enough that `log` stays finite.

`clamp` is exposed as a free function, built the same way as `relu` or `sigmoid`, but it needs extra parameters (`min_val`, `max_val`), so it wraps the forward/backward in small lambdas before handing them to `ApplyUnaryOp`:

```c++
// core/ops.cc (Version 3)
Tensor clamp(const Tensor& a, float min_val, float max_val) {
  auto forward_op = [min_val, max_val](const std::vector<float>& x) {
    return ClampForward(x, min_val, max_val);
  };
  auto backward_op = [min_val, max_val](Tensor& out, const Tensor& x) {
    ClampBackward(out, x, min_val, max_val);
  };
  return Tensor::ApplyUnaryOp(a, "Clamp", forward_op, backward_op);
}
```

#### `backward()` requires a scalar — and the bug it uncovered

Up until now, `Tensor::backward()` simply seeded *every* element of the output's gradient with `1.0` and started backpropagating. But that's only mathematically valid if the output is a **scalar** ($1$ element) — the chain rule's starting point $\frac{\partial L}{\partial L} = 1$ assumes $L$ is a single number. Calling `backward()` on a non-scalar tensor (say, a $(4,4)$ tensor) doesn't have a well-defined meaning and was a silent footgun.

Version 3 makes this an explicit, loud error:

```c++
// core/tensor.cc (Version 3)
void Tensor::backward() {
  if (!node_->requires_grad) return;
  if (size() != 1) {
    throw std::runtime_error("backward() can only be called on a scalar (size 1) tensor");
  }
  node_->EnsureGrad();
  std::fill(node_->grad.begin(), node_->grad.end(), 1.0f);
  ...
}
```

Adding this check immediately broke the `Sum` and `Mean` gradient-check tests — and in the process of debugging *that*, a real bug in `ApplyUnaryOp` came to light. Recall that `sum(x)` and `mean(x)` take a tensor of any shape and reduce it to a single number. But `ApplyUnaryOp` was unconditionally giving the output node the *same shape as the input*:

```c++
// Before (Version 2): always wrong for reductions!
std::vector<int> out_shape = a.shape();
```

So `sum(x)` on a $(2,2)$ input would produce a `ComputeNode` whose `data` had 1 element (the sum) but whose `shape` claimed `{2, 2}` — a 4-element shape for a 1-element tensor! `size()` would then (incorrectly) report `4`, tripping the new scalar check on a perfectly valid `sum().backward()` call. The fix:

```c++
// core/tensor.cc (Version 3)
Tensor Tensor::ApplyUnaryOp(const Tensor& a,
                            const std::string& op_name,
                            UnaryMathOp forward_op,
                            UnaryGradOp backward_op) {
  std::vector<float> out_data = forward_op(a.data());
  // Elementwise ops preserve the input shape; reductions (e.g. sum, mean)
  // collapse to a flat 1-element tensor.
  std::vector<int> out_shape = (out_data.size() == a.data().size())
                                    ? a.shape()
                                    : std::vector<int>{static_cast<int>(out_data.size())};
  ...
}
```

Now the output shape is derived from the *actual size of the data the forward pass produced* — `{2,2}` for elementwise ops like `relu`, `{1}` for reductions like `sum`/`mean`. This is a great example of why gradient checking earns its place in the test suite: the `backward()` scalar check and the `ApplyUnaryOp` shape bug were two separate issues, and tightening the first immediately exposed the second.

#### Division by zero: fail loudly, not silently

The last fix (currently sitting as an uncommitted change on top of everything above) closes a gap in `DivForward`. Dividing by zero in IEEE floating point doesn't crash — it quietly produces `inf` or `NaN`, which then flows through the rest of the graph, multiplying and adding its way into every downstream value and gradient. By the time you notice your loss is `NaN`, you have no idea which division caused it.

```c++
// core/ops.cc (Version 3, uncommitted)

// Throws if any element of the divisor is zero, instead of silently
// producing inf/NaN that then propagates through forward and backward.
void CheckNoZeroDivisor(const std::vector<float>& b) {
  for (float val : b) {
    if (val == 0.0f) {
      throw std::invalid_argument("division by zero: divisor tensor contains a zero element");
    }
  }
}

std::vector<float> DivForward(const std::vector<float>& a,
                              const std::vector<float>& b,
                              const std::vector<int>& shape_a,
                              const std::vector<int>& shape_b) {
  CheckNoZeroDivisor(b);
  ...
}
```

Now `a / b` throws `std::invalid_argument` immediately, at the point of the mistake, with a message that tells you exactly what went wrong — instead of a baffling `NaN` three layers downstream. The accompanying test in `tests/test_gradcheck.cc` confirms this:

```c++
// tests/test_gradcheck.cc
bool threw = false;
try {
  Tensor a({1.0f, 2.0f}, {2});
  Tensor b({1.0f, 0.0f}, {2});
  [[maybe_unused]] Tensor c = a / b;
} catch (const std::invalid_argument&) {
  threw = true;
}
```

### Continuous Integration: never trust "it worked on my machine"

With a real test suite in place, it would be a shame to rely on remembering to run `make test` before every push. Version 3 adds a tiny GitHub Actions workflow:

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
  pull_request:

jobs:
  build-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build
        run: make

      - name: Run tests
        run: make test
```

Every push and every pull request now automatically builds the project and runs the full gradient-check suite on a clean Ubuntu machine. If a future change reintroduces a sign error in `MulBackward`, or breaks `clamp`, CI catches it before it ever reaches `main` — no more "but it worked on my machine".

### What can we build now?

Let's circle back to `TrainMLP` in `main.cc` — the XOR problem, the "Hello World" of neural networks precisely *because* it's not linearly separable; a single-layer model can't solve it, you need a hidden layer with a non-linearity.

```c++
// main.cc
Adam optimizer({W1, b1, W2, b2}, 0.01f);
MSELoss criterion;

for (int epoch = 0; epoch < 1000; ++epoch) {
  optimizer.zero_grad();

  // Forward Pass
  Tensor h_pre = matmul(X, W1) + b1; // Broadcasting b1 here!
  Tensor h = relu(h_pre); 
  
  Tensor out = matmul(h, W2) + b2;   // Broadcasting b2 here!
  Tensor loss = criterion(out, Y);

  loss.backward();
  optimizer.step();
}
```

Compare this to Version 0's training loop, where we manually wrote `w.mutable_data() -= learning_rate * w.grad()` for two scalar parameters, inside a hand-rolled loop over every data point. Here, in roughly a dozen lines:

- `X` and `Y` hold the *entire* dataset as batched tensors.
- `matmul` and broadcasting `+` build a real two-layer MLP forward pass.
- `relu` gives us the non-linearity needed to solve XOR.
- `MSELoss` turns predictions and targets into a single scalar loss.
- `Adam` adapts the learning rate per-parameter and updates all four tensors (`W1`, `b1`, `W2`, `b2`) in one `step()` call.
- `zero_grad()` resets gradients cleanly every epoch — no manual bookkeeping.
- Under the hood, every `ComputeNode` is pulled from a memory pool, every op's lambda is as small as it can be, `clamp` keeps `log`-based losses finite, and a CI pipeline runs gradient checks on every push to make sure none of this silently breaks.

None of this was possible at the end of Version 2 — broadcasting alone got us *part* of the way, but without optimizers, losses, and the bug fixes above, this loop would either not compile, not converge, or quietly produce `NaN`. mini-autodiff has gone from a toy that proves a concept to a small but genuinely *usable* tool.