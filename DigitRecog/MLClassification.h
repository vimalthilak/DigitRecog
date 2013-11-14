#include "Services/Messaging.h"
#include "Services/IIncidentListener.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <utility>      // std::pair
#include <mutex>

class IRootNtupleWriterTool;
class CvRTParams;

namespace cv {
  class Mat;
}

class MLClassification: public Messaging, virtual public IIncidentListener
{
public:
 

  MLClassification(const std::string& n);
  MLClassification(const std::string& n, TLogLevel);
  ~MLClassification();
  
  
  
  //1st arg: target
  //2nd arg: vector of attributes
  void accumulateTrain(int, const std::vector<float>&);
  void accumulateTest(int, const std::vector<float>&);
  
  void performCrossValidationTraining(unsigned int,
                                      int max_depth, int min_sample, int num_var,
                                      int num_trees=500);
  
  void setRootNtupleHelper(IRootNtupleWriterTool * h) { m_ntuple_helper = h; }
    
  //IIncidentListener impl
  virtual void handle(const Incident&);
  
protected:

  int initialize();
  
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
  
  void accumulate(int, const std::vector<float>&, cv::Mat*&, cv::Mat*&);
  void scale(cv::Mat*);
  
  std::mutex m_log_mtx;
  
  void train(char&, const int , const CvRTParams* ,
           const std::vector< std::vector<int> >&, std::unordered_map<unsigned int, std::pair<unsigned char,unsigned char> > &,
           std::unordered_map<unsigned int, std::map< unsigned char, float > > &);
  
  
private:

   void Register();

};
