# include "Qdefs.hpp"
# include <iostream>

class Model {
  public:
    Model(): x("hello") {}

    char x[10];
};

class Some {
  public:
    Some() {}
    const char* show(Model* model) {return model->x;}
};
