#include <dmlc/logging.h>

int main(void) {
  LOG(INFO) << "hello";
  LOG(ERROR) << "error";
  try {
    LOG(FATAL)<<'a'<<11<<33;
  } catch (dmlc::Error e) {
    LOG(INFO) << "catch " << e.what();
  }
  CHECK(2!=3) << "test";
  CHECK(2==3) << "test";
  return 0;
}
