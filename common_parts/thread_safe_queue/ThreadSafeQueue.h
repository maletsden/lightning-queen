#ifndef COMMON_PARTS_THREAD_SAFE_QUEUE_H
#define COMMON_PARTS_THREAD_SAFE_QUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>

template <class T>
class ThreadSafeQueue
{
public:
  ThreadSafeQueue()
      : q()
      , m()
      , c()
  {}

  ~ThreadSafeQueue() = default;

  // Add an element to the queue.
  template<class D>
  void enqueue(D&& t)
  {
    std::lock_guard<std::mutex> lock(m);
    q.push(std::forward<D>(t));
    c.notify_one();
  }

  // Get the "front"-element.
  // If the queue is empty, wait till a element is available.
  T dequeue()
  {
    std::unique_lock<std::mutex> lock(m);
    while(q.empty())
    {
      // release lock as long as the wait and reacquire it afterwards.
      c.wait(lock);
    }
    T val = q.front();
    q.pop();
    return val;
  }

private:
  std::queue<T> q;
  mutable std::mutex m;
  std::condition_variable c;
};

#endif //COMMON_PARTS_THREAD_SAFE_QUEUE_H
