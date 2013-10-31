#ifndef IPREPROCESSORTOOLS_H
#define IPREPROCESSORTOOLS_H

#include <vector>

class IPreProcessorTools {



public:
   virtual void execute(std::vector<float>&) = 0;
   
   virtual ~IPreProcessorTools() {};

};

#endif