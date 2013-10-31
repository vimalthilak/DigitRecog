%module DigitRecog
%{
#include "DigitRecog/CannyEdgeTool.h"
#include "DigitRecog/GaussianBlurTool.h"
#include "DigitRecog/FeX.h"
%}

%include "std_string.i"
%include "std_vector.i"

namespace std {
       %template(FloatVector) vector<float>;
}

class IPreProcessorTools {

public:

   virtual ~IPreProcessorTools() ;
   
protected:
   
   IPreProcessorTools();
}; 


class CannyEdgeTool : virtual public IPreProcessorTools {
    
    
public:
    CannyEdgeTool();
    ~CannyEdgeTool();

    
    float GetLowHystThreshold2();
    float GetHighHystThreshold2();
    
    void SetLowHystThreshold2(float);
    void SetHighHystThreshold2(float);
};

class GaussianBlurTool : virtual public IPreProcessorTools {
    
    
public:
    GaussianBlurTool() {};
    ~GaussianBlurTool() {};

};

enum TLogLevel  {logERROR, logWARNING, logINFO, logDEBUG, logVERBOSE};

class FeX
{
public:
 

  FeX(const std::string&, IRootNtupleWriterTool *, IRootNtupleWriterTool *);
  FeX(const std::string&, TLogLevel, IRootNtupleWriterTool *, IRootNtupleWriterTool *);
  
  
  int execute(int, std::vector<float>&);
  
  void addPreProcessorTool(IPreProcessorTools*);
};
