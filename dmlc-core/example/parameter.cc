// This is an example program showing usage of parameter module
// Build, on root folder, type
//
//   make example
//
// Example usage:
//
//   example/parameter num_hidden=100 name=aaa activation=relu
//

#include <dmlc/parameter.h>

struct MyParam : public dmlc::Parameter<MyParam> {
  float learning_rate;
  int num_hidden;
  int activation;
  std::string name;
  // declare parameters in header file
  DMLC_DECLARE_PARAMETER(MyParam) {
    DMLC_DECLARE_FIELD(num_hidden).set_range(0, 1000)
        .describe("Number of hidden unit in the fully connected layer.");
    DMLC_DECLARE_FIELD(learning_rate).set_default(0.01f)
        .describe("Learning rate of SGD optimization.");
    DMLC_DECLARE_FIELD(activation).add_enum("relu", 1).add_enum("sigmoid", 2)
        .describe("Activation function type.");
    DMLC_DECLARE_FIELD(name).set_default("mnet")
        .describe("Name of the net.");

    // user can also set nhidden besides num_hidden
    DMLC_DECLARE_ALIAS(num_hidden, nhidden);
    DMLC_DECLARE_ALIAS(activation, act);
  }
};

// register it in cc file
DMLC_REGISTER_PARAMETER(MyParam);


int main(int argc, char *argv[]) {
  if (argc == 1) {
    printf("Usage: [key=value] ...\n");
    return 0;
  }

  MyParam param;
  std::map<std::string, std::string> kwargs;
  for (int i = 0; i < argc; ++i) {
    char name[256], val[256];
    if (sscanf(argv[i], "%[^=]=%[^\n]", name, val) == 2) {
      kwargs[name] = val;
    }
  }
  printf("Docstring\n---------\n%s", MyParam::__DOC__().c_str());

  printf("start to set parameters ...\n");
  param.Init(kwargs);
  printf("-----\n");
  printf("param.num_hidden=%d\n", param.num_hidden);
  printf("param.learning_rate=%f\n", param.learning_rate);
  printf("param.name=%s\n", param.name.c_str());
  printf("param.activation=%d\n", param.activation);
  return 0;
}

