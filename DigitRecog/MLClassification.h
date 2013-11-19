#include "Services/Messaging.h"
#include "Services/IIncidentListener.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <utility>      // std::pair
#include <mutex>
#include <atomic>
#include <algorithm>

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
  void accumulateTrain(int, const std::vector<double>&);
  void accumulateTest(int, const std::vector<double>&);
  
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
  
  std::vector<double> m_training_max;
  std::vector<double> m_training_min;
    
  IRootNtupleWriterTool * m_ntuple_helper;
  
  std::size_t m_nvar;
  
  void init(std::size_t);
    
  template <class T>
  void pushBack(const std::string & , const T*, IRootNtupleWriterTool *);
  
  void accumulate(int, const std::vector<double>&, cv::Mat*&, cv::Mat*&);
  void scale(cv::Mat*);
  
  std::mutex m_log_mtx;
  
  void train(char&, const int , std::atomic<int>&, const CvRTParams* ,
           const std::vector< std::vector<int> >&, std::unordered_map<unsigned int, std::pair<unsigned char,unsigned char> > &,
           std::unordered_map<unsigned int, std::map< unsigned char, float > > &);
  
  
  //helper methods
  template<class T1, class T2>
  struct pairCompare {
  bool operator() (const std::pair<T1,T2> & x, const std::pair<T1,T2> & y)
                   {
                      return x.second < y.second;
                   }
  };
    
  template<class T>
  typename T::const_iterator map_max_element(const T & A)
  {
      typedef typename T::value_type pair_type;
      typedef typename pair_type::first_type K;
      typedef typename pair_type::second_type V;
      return std::max_element(A.cbegin(), A.cend(), pairCompare<K,V>());
  }
  
  
private:

   void Register();

};












