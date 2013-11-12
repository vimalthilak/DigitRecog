#include "DigitRecog/MLClassification.h"
#include "UtilityToolsInterfaces/IRootNtupleWriterTool.h"
#include "UtilityToolsInterfaces/SingleObjectHolder.h"
#include "UtilityToolsInterfaces/VectorObjectHolder.h"
#include "Services/IncidentService.h"



//opencv huge include file
#include "opencv/ml.h"


//c++11
#include <random>
#include <thread>
#include <mutex>

#include <cmath>
#include <utility>      // std::pair
#include <set>


//helper macros
#define REGBRANCH_S(name, type, initVal) m_ntuple_helper->registerBranch(#name, new SingleObjectHolder< type >(initVal));
#define REGBRANCH_V(name, type)          m_ntuple_helper->registerBranch(#name, new VectorObjectHolder< type >());
#define PUSHBACK(name, obj)              pushBack(#name, &(obj), m_ntuple_helper );


// Constructor
///////////////////////////
MLClassification::MLClassification(const std::string& n, IRootNtupleWriterTool* r):
        Messaging(n), m_training_data(0), m_training_classes(0), m_var_type(0),
                      m_testing_data(0), m_testing_classes(0),
        m_ntuple_helper(r), m_nvar(0) {
    
    Register();
    
}

///////////////////////////
MLClassification::MLClassification(const std::string& n, TLogLevel m_lvl, IRootNtupleWriterTool* r):
         Messaging(n, m_lvl), m_training_data(0), m_training_classes(0), m_var_type(0),
                      m_testing_data(0), m_testing_classes(0),
        m_ntuple_helper(r), m_nvar(0) {
    
    Register();
    
}

MLClassification::~MLClassification() {

  if (m_training_data) delete m_training_data;
  if (m_training_classes) delete m_training_classes;
  if (m_var_type) delete m_var_type;
  if (m_testing_data) delete m_testing_data;
  if (m_testing_classes) delete m_testing_classes;

}

void MLClassification::Register() {

    // Get handle on IncidentSvc
    IncidentService * inc_svc = IncidentService::getInstance();
    if (!inc_svc) {LOG("Coulnd't get IncidentService", logERROR); return; }
    
    inc_svc->addListener(this, "BeginRun");

}

// Initialization
////////////////////////////
int MLClassification::initialize()
{
    
    LOG("Initialization..",logDEBUG);
    
    REGBRANCH_S(target, int, -1)
    
    REGBRANCH_S(prediction, int, -1)

    REGBRANCH_V(prediction_prob, float)
    
    
    
    return 1;
    
}

void MLClassification::init(std::size_t s_) {

   if (m_nvar == 0 ) m_nvar = s_;
   if (!m_var_type) {
       m_var_type = new cv::Mat(s_+1, 1, CV_8U, cv::Scalar(CV_VAR_NUMERICAL));
       m_var_type->at<uchar>(s_, 0) = CV_VAR_CATEGORICAL; //classes
   }

}

// Accumulate
//////////////////////////////
void MLClassification::accumulate(int target, const std::vector<float>& features, cv::Mat * data, cv::Mat * classes) {
    
    
    if (!data) {
        
        init(features.size());
        
        data = new cv::Mat(1, m_nvar, cv::DataType<float>::type); //floats
        classes = new cv::Mat(1, 1, cv::DataType<int>::type); //integers
        
        data->reserve(10000);
        classes->reserve(10000);
        
    }
    
    if (m_nvar != features.size()) {LOG("Mismatch in features vector size.. got "<<
                                        features.size()<<", expecting: "<<m_nvar, logERROR); return;}
    
    cv::Mat input(features, false); //copydata=false
    
    data->push_back(input.t()); //transpose so we have one row with m_nvar columns
    
    classes->push_back(target);
    
}


// Accumulate - Training
//////////////////////////////
void MLClassification::accumulateTrain(int target, const std::vector<float>& features) {

  accumulate(target, features, m_training_data, m_training_classes);
  
  if (m_training_max.empty()) {
      LOG("Preparing max/min vectors..", logDEBUG);
      m_training_max.resize(m_nvar, -1.e99);
      m_training_min.resize(m_nvar, 1.e99);
  }
  
  for (std::size_t it_val = 0; it_val < m_nvar; ++it_val) {
      float val = features.at(it_val);
      if (val > m_training_max.at(it_val)) m_training_max[it_val] = val;
      if (val < m_training_min.at(it_val)) m_training_min[it_val] = val;
  }
  
}

// Accumulate - Testing
//////////////////////////////
void MLClassification::accumulateTest(int target, const std::vector<float>& features) {

  accumulate(target, features, m_testing_data, m_testing_classes);
  
}


void MLClassification::scale(cv::Mat* data) {

   //sanity
   if ((int)m_nvar != data->size().width) {LOG("Prob, width of matrix mismatch", logERROR); return; }
   if ( m_training_max.empty() || m_training_min.empty() || m_training_max.size() != m_training_min.size() || m_training_max.size() != m_nvar) {
      LOG("Prob with max/min vectors", logERROR); return;
   }

   for (int i_values = 0; i_values < data->size().height; ++i_values) {
        for (int i =0; i < data->size().width; ++i) {
             data->at<float>(i_values,i) = ( (data->at<float>(i_values,i) - m_training_min.at(i))/(m_training_max.at(i)-m_training_min.at(i)))*2. + -1.;  //[-1, +1]
        }
    }

}

// CV training
///////////////////////////
void MLClassification::performCrossValidationTraining(unsigned int n_regions) {

    //scale data --> [-1, 1]
    scale(m_training_data);
    
    //randomly seperate sample into four equal parts
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0,n_regions-1); // generates number in the range 0..n_regions-1
    auto dice = std::bind ( distribution, generator );
    
    std::vector<std::vector<int> > cv_indices;
    cv_indices.resize(n_regions);
    for (int i = 0; i < m_training_data->size().height; ++i) { 
        int num = dice();
        cv_indices[num].push_back(i);
    }
    
    LOG("sizes of cross_validation regions: "<<cv_indices[1].size()<<", "<<cv_indices[2].size()<<", "<<cv_indices[3].size()<<", "<<cv_indices[4].size(), logDEBUG);
    
    
    
    
   




}


////////////////////////////////////////////////////////////////////////////////
// Handle incidents
////////////////////////////////////////////////////////////////////////////////
void MLClassification::handle(const Incident& inc)
{
    
    if (inc.svcType() == "BeginRun") {
        
        if ( !(this->initialize())) { LOG("Couldn't initialize properly..", logERROR); }
        
    } //else if
    
    
    
} //void

template <class T>
inline void MLClassification::pushBack(const std::string & branch_name, const T* obj, IRootNtupleWriterTool * m_ntuple) {
    m_ntuple->pushBack(branch_name, boost::any(obj));
}


