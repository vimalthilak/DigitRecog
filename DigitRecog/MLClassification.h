#include "Services/Messaging.h"
#include "Services/IIncidentListener.h"
#include <string>
#include <vector>

class IRootNtupleWriterTool;

namespace cv {
  class Mat;
}

class MLClassification: public Messaging, virtual public IIncidentListener
{
public:
 

  MLClassification(const std::string& n, IRootNtupleWriterTool* );
  MLClassification(const std::string& n, TLogLevel, IRootNtupleWriterTool *);
  ~MLClassification();
  
  int initialize();
  
  //1st arg: target
  //2nd arg: vector of attributes
  void accumulateTrain(int, const std::vector<float>&);
  void accumulateTest(int, const std::vector<float>&);
  
  void performCrossValidationTraining(unsigned int);
  
    
  //IIncidentListener impl
  virtual void handle(const Incident&);
  
protected:

  cv::Mat * m_training_data;
  cv::Mat * m_training_classes;
    
  cv::Mat * m_var_type;
  
  cv::Mat * m_testing_data;
  cv::Mat * m_testing_classes;
  
  std::vector<float> m_training_max;
  std::vector<float> m_training_min;
    

  IRootNtupleWriterTool * m_ntuple_helper;
  
  std::size_t m_nvar;
  
  void init(std::size_t);
    
  template <class T>
  void pushBack(const std::string & , const T*, IRootNtupleWriterTool *);
  
  void accumulate(int, const std::vector<float>&, cv::Mat*, cv::Mat*);
  void scale(cv::Mat*);
  
  
private:

   void Register();

};
