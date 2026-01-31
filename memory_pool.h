#ifndef AUTOGRAD_MEMORY_POOL_H_
#define AUTOGRAD_MEMORY_POOL_H_

#include <vector>
#include <mutex>
#include <memory>
#include <cassert>

namespace autograd {

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

template <class T, class U>
bool operator==(const PoolAllocator<T>&, const PoolAllocator<U>&) { return true; }
template <class T, class U>
bool operator!=(const PoolAllocator<T>&, const PoolAllocator<U>&) { return false; }

}  // namespace autograd

#endif  // AUTOGRAD_MEMORY_POOL_H_