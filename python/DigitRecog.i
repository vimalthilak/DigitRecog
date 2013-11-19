%module DigitRecog
%{
#include "DigitRecog/CannyEdgeTool.h"
#include "DigitRecog/GaussianBlurTool.h"
#include "DigitRecog/FeX.h"
#include "DigitRecog/MLClassification.h"
%}

%include "exception.i"

%exception {
    try {
        $action
    }
    catch (const std::exception & e)
    {
        SWIG_exception(SWIG_RuntimeError, (std::string("C++ std::exception in $decl: ") + e.what()).c_str());
    }
    catch (...)
    {
        SWIG_exception(SWIG_UnknownError, "C++ anonymous exception");
    }
}


%include "std_string.i"
%include "std_vector.i"

namespace std {
       %template(DoubleVector) vector<double>;
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



class MLClassification 
{

public:
   MLClassification(const std::string& n);
   MLClassification(const std::string& n, TLogLevel);

   void accumulateTrain(int, const std::vector<double>&);
   void accumulateTest(int, const std::vector<double>&);
  
   void performCrossValidationTraining(unsigned int,
                                      int max_depth, int min_sample, int num_var,
                                      int num_trees=500);
   
   void performTraining(int max_depth, int min_sample, int num_var,
                        int num_trees=500);
        
   void performTesting(const std::string&);
                             
   void setRootNtupleHelper(IRootNtupleWriterTool *);


};










