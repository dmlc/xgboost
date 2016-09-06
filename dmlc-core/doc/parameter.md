Parameter Structure for Machine Learning
========================================
One of the most important ingredients of machine learning projects are the parameters.
Parameters act as a way of communication between users and the library. In this article, we will introduce the parameter module of DMLC, a lightweight C++ module that is designed to support
general machine learning libraries. It comes with the following nice properties:

- Easy declaration of typed fields, default values and constraints.
- Auto checking of constraints and throw exceptions when constraint is not met.
- Auto generation of human readable docstrings on parameters.
- Serialization and de-serialization into JSON and ```std::map<std::string, std::string>```.

Use Parameter Module
--------------------
### Declare the Parameter
In the dmlc parameter module, every parameter can be declared as a structure. 
This means you can easily access these fields as they normally are efficiently.
For example, it is very common to write 
```c++
weight -= param.learning_rate * gradient;
```

The only difference between a normal structure is that we will need to declare
all the fields, as well as their default value and constraints.
The following code gives an example of declaring parameter structure ```MyParam```.

```c++
#include <dmlc/parameter.h>

// declare the parameter, normally put it in header file.
struct MyParam : public dmlc::Parameter<MyParam> {
  float learning_rate;
  int num_hidden;
  int activation;
  std::string name;
  // declare parameters
  DMLC_DECLARE_PARAMETER(MyParam) {
    DMLC_DECLARE_FIELD(num_hidden).set_range(0, 1000)
        .describe("Number of hidden unit in the fully connected layer.");
    DMLC_DECLARE_FIELD(learning_rate).set_default(0.01f)
        .describe("Learning rate of SGD optimization.");
    DMLC_DECLARE_FIELD(activation).add_enum("relu", 1).add_enum("sigmoid", 2)
        .describe("Activation function type.");
    DMLC_DECLARE_FIELD(name).set_default("layer")
        .describe("Name of the net.");
  }
};

// register the parameter, this is normally in a cc file.
DMLC_REGISTER_PARAMETER(MyParam);
```

We can find that the only difference is the lines after ```DMLC_DECLARE_PARAMETER(MyParam)```,
where all the fields are declared. In this example, we have declared parameters of ```float,int,string``` types.
Here are some highlights in this example:

- For the numeric parameters, it is possible to set a range constraints via ```.set_range(begin, end)```.
- It is possible to define enumeration types, in this case activation. 
  User is only allowed to set ```sigmoid``` or ```relu``` into the activation field, and they will be mapped into 1 and 2 separately.
- The ```describe``` function adds a description on the field, which is used to generate human readable docstring. 

### Set the Parameters
After we declared the parameters, we can declare this structure as normal structure.
Except that the ```MyParam``` structure now comes with a few member functions 
to make parameter manipulation easy.
To set the parameters from external data source, we can use the ```Init``` function.
```c++
int main() {
   MyParam param;
   std::vector<std::pair<std::string, std::string> > param_data = {
     {"num_hidden", "100"},
	 {"activation", "relu"},
	 {"name", "myname"}
   };
   // set the parameters
   param.Init(param_data);
   return 0;
}
```
After the ```Init``` function is called, the ```param``` will be filled with the specified key values in ```param_data```.
More importantly, the ```Init``` function will do automatic checking of parameter range and throw an ```dmlc::ParamError``` 
with detailed error message if things went wrong.

### Generate Human Readable Docstrings
Another useful feature of the parameter module is to get an human readable docstring of the parameter.
This is helpful when we are creating language binding such as python and R, and we can use it to generate docstring of 
foreign language interface.

The following code obtains the dostring of ```MyParam```.
```c++
std::string docstring = MyParam::__DOC__();
```

We also provide a more structured way to access the detail of the fields(name, default value, detailed description) via
```c++
std::vector<dmlc::ParamFieldInfo> fields = MyParam::__FIELDS__();
```

### Serialization of Parameters
One of the most common way to serialize the parameter is to convert it back to representation of ```std::map<string, string>```
by using the following code. 
```c++
std::map<string, string> dict = param.__DICT__();
```
The ```std::map<string, string>``` can further be serialized easily. This way of serialization is more device and platform(32/64 bit) agnostic.
However, this is not very compact, and recommended only used to serialize the general parameters set by the user.

Direct serialization and loading of JSON format is also support.

### Play with an Example
We provide an example program [parameter.cc](https://github.com/dmlc/dmlc-core/blob/master/example/parameter.cc), to 
demonstrate the usage mentioned above, and allow you to play with it and get sense of what is going on.

How does it work
----------------
Hope you like the parameter module so far. In this section, we will explain how does it work. Making such parameter module 
in ```C++``` is not easy. Because this basically means some way of reflection -- getting the information of fields in a 
structure out, which is not supported by ```C++```. 

Consider the following program, how do the Init function know the location of ```num_hidden```, and set it correctly
in ```Init``` function?

```c++
#include <vector>
#include <string>
#include <dmlc/parameter.h>

// declare the parameter, normally put it in header file.
struct MyParam : public dmlc::Parameter<MyParam> {
  float learning_rate;
  int num_hidden;
  // declare parameters
  DMLC_DECLARE_PARAMETER(MyParam) {
    DMLC_DECLARE_FIELD(num_hidden);
    DMLC_DECLARE_FIELD(learning_rate).set_default(0.01f);
  }
};

// register the parameter, this is normally in a cc file.
DMLC_REGISTER_PARAMETER(MyParam);

int main(int argc, char *argv[]) {
  MyParam param;
  std::vector<std::pair<std::string, std::string> > param_data = {
    {"num_hidden", "100"},
  };
  param.Init(param_data);
  return 0;
}
```

The secrete lies in the function ```DMLC_DECLARE_PARAMETER(MyParam)```, this is an macro defined in the parameter module.
If we expand the micro, the code roughly becomes the following code.

```c++
struct Parameter<MyParam> {
  template<typename ValueType>
  inline FieldEntry<ValueType>&
  DECLARE(ParamManagerSingleton<MyParam> *manager,
		  const std::string& key,
		  ValueType& ref){
	// offset gives a generic way to access the address of the field
	// from beginning of the structure.
	size_t offset = ((char*)&ref - (char*)this);
	parameter::FieldEntry<ValueType> *e =
        new parameter::FieldEntry<ValueType>(key, offset);
	manager->AddEntry(key, e);
	return *e;
  }
};

struct MyParam : public dmlc::Parameter<MyParam> {
  float learning_rate;
  int num_hidden;
  // declare parameters
  inline void __DECLARE__(ParamManagerSingleton<MyParam> *manager) {
    this->DECLARE(manager, "num_hidden", num_hidden);
	this->DECLARE(manager, "learning_rate", learning_rate).set_default(0.01f);
  }
};

// This code is only used to show the general idea.
// This code will only run once, the real code is done via singleton declaration pattern.
{
  static ParamManagerSingleton<MyParam> manager;
  MyParam tmp;
  tmp->__DECLARE__(&manager);
}
```
This is not the actual code that runs, but generally shows the idea on how it works. 
The key is that the structure layout is fixed for all the instances of objects.
To figure out how to access each of the field, we can
- Create an instance of MyParam, call the ```__DECLARE__``` function.
- The relative position of the field against the head of the structure is recorded into a global singleton.
- When we call ```Init```, we can get the ```offset``` from the singleton, and access the address of the field via ```(ValueType*)((char*)this + offset)```.

You are welcomed to check out the real details in [dmlc/parameter.h](https://github.com/dmlc/dmlc-core/blob/master/include/dmlc/parameter.h).
By using the generic template programming in C++, we have created a simple and useful parameter module for machine learning libraries.
This module is used extensively by DMLC projects. Hope you will find it useful as well :).
