#include <rabit/rabit.h>
using namespace rabit;
const int N = 3;
int main(int argc, char *argv[]) {
  rabit::Init(argc, argv);
  std::string s;
  if (rabit::GetRank() == 0) s = "hello world";
  printf("@node[%d] before-broadcast: s=\"%s\"\n",
         rabit::GetRank(), s.c_str());
  // broadcast s from node 0 to all other nodes
  rabit::Broadcast(&s, 0);
  printf("@node[%d] after-broadcast: s=\"%s\"\n",
         rabit::GetRank(), s.c_str());
  rabit::Finalize();
  return 0;
}
