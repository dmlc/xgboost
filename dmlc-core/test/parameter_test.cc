#include <dmlc/parameter.h>

// this is actual pice of code
struct Param : public dmlc::Parameter<Param> {
  float learning_rate;
  int num_hidden;
  int act;
  std::string name;
  // declare parameters in header file
  DMLC_DECLARE_PARAMETER(Param) {
    DMLC_DECLARE_FIELD(num_hidden).set_range(0, 1000)
        .describe("Number of hidden unit in the fully connected layer.");
    DMLC_DECLARE_FIELD(learning_rate).set_default(0.01f)
        .describe("Learning rate of SGD optimization.");
    DMLC_DECLARE_FIELD(act).add_enum("relu", 1).add_enum("sigmoid", 2)
        .describe("Activation function type.");
    DMLC_DECLARE_FIELD(name).set_default("A")
        .describe("Name of the net.");
  }
};

// this is actual pice of code
struct SecondParam : public dmlc::Parameter<SecondParam> {
  int num_data;
  // declare parameters in header file
  DMLC_DECLARE_PARAMETER(SecondParam) {
    DMLC_DECLARE_FIELD(num_data).set_range(0, 1000)
        .describe("Number of data points");
  }
};
// register it in cc file
DMLC_REGISTER_PARAMETER(Param);
DMLC_REGISTER_PARAMETER(SecondParam);

int main(int argc, char *argv[]) {
  Param param;
  SecondParam param2;
  std::map<std::string, std::string> kwargs;
  for (int i = 0; i < argc; ++i) {
    char name[256], val[256];
    if (sscanf(argv[i], "%[^=]=%[^\n]", name, val) == 2) {
      printf("call set %s=%s\n", name, val);
      kwargs[name] = val;
    }
  }
  printf("Parameters\n-----------\n%s", Param::__DOC__().c_str());
  std::vector<std::pair<std::string, std::string> > unknown;
  unknown = param.InitAllowUnknown(kwargs);
  unknown = param2.InitAllowUnknown(unknown);

  if (unknown.size() != 0) {
    std::ostringstream os;
    os << "Cannot find argument \'" << unknown[0].first << "\', Possible Arguments:\n";
    os << "----------------\n";
    os << param.__DOC__();
    os << param2.__DOC__();
    throw dmlc::ParamError(os.str());
  }
  printf("-----\n");
  printf("param.num_hidden=%d\n", param.num_hidden);
  printf("param.learning_rate=%f\n", param.learning_rate);
  printf("param.name=%s\n", param.name.c_str());
  printf("param.act=%d\n", param.act);
  printf("param.size=%lu\n", sizeof(param));

  printf("Unknown parameters:\n");
  for (size_t i = 0; i < unknown.size(); ++i) {
    printf("%s=%s\n", unknown[i].first.c_str(), unknown[i].second.c_str());
  }

  std::ostringstream os;
  dmlc::JSONWriter writer(&os);
  param.Save(&writer);
  printf("JSON:\n%s\n", os.str().c_str());
  printf("Environment variables\n");
  int test_env = dmlc::GetEnv("TEST_ENV", 1);
  std::string test_env2 = dmlc::GetEnv<std::string>("TEST_ENV2", "hello");
  printf("TEST_ENV=%d\n", test_env);
  printf("TEST_ENV2=%s\n", test_env2.c_str());

  return 0;
}
