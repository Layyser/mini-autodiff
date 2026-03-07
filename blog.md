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