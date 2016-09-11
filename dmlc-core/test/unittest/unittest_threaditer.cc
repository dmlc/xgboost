#include <dmlc/io.h>
#include <dmlc/logging.h>
#include <chrono>
#include <gtest/gtest.h>
#include <dmlc/threadediter.h>

using namespace dmlc;
namespace producer_test {
inline void delay(int sleep) {
  if (sleep < 0) {
    int d = rand() % (-sleep);
    std::this_thread::sleep_for(std::chrono::milliseconds(d));
  } else {
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep));
  }
}

// int was only used as example, in real life
// use big data blob
struct IntProducer : public ThreadedIter<int>::Producer {
  int counter;
  int maxcap;
  int sleep;
  IntProducer(int maxcap, int sleep)
      : counter(0), maxcap(maxcap), sleep(sleep) {}
  virtual void BeforeFirst(void) {
    counter = 0;
  }
  virtual bool Next(int **inout_dptr) {
    if (counter == maxcap) return false;
    // allocate space if not exist
    if (*inout_dptr == NULL) {
      *inout_dptr = new int();
    }
    delay(sleep);
    **inout_dptr = counter++;
    return true;
  }
};

}

TEST(ThreadedIter, basics) {
  using namespace producer_test;
  ThreadedIter<int> iter;
  iter.set_max_capacity(1);
  IntProducer prod(10, 100);
  int d = 100;
  iter.Init(&prod);
  int counter = 0;
  while (iter.Next()) {
    CHECK(counter == iter.Value());
    delay(d);
    LOG(INFO)  << counter;
    ++counter;
  }
  CHECK(!iter.Next());
  iter.BeforeFirst();
  iter.BeforeFirst();
  iter.BeforeFirst();
  iter.Next();
  iter.BeforeFirst();
  iter.BeforeFirst();
  counter = 0;
  int *value;
  while (iter.Next(&value)) {
    LOG(INFO)  << *value;
    CHECK(counter == *value);
    ++counter;
    iter.Recycle(&value);
    delay(d);
    CHECK(value == NULL);
  }
  LOG(INFO) << "finish";
}
