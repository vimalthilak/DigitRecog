#include "DigitRecog/MLClassification.h"
#include "UtilityToolsInterfaces/IRootNtupleWriterTool.h"
#include "UtilityToolsInterfaces/SingleObjectHolder.h"
#include "UtilityToolsInterfaces/VectorObjectHolder.h"
#include "Services/IncidentService.h"
#include "DigitRecog/IPreProcessorTools.h"
#include <cmath>
#include <utility>      // std::pair
#include <set>

//opencv huge include file
#include "opencv/ml.h"

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
    
    //REGBRANCH_S(target, int, -1)
    
    //REGBRANCH_V(features, float)

    //REGBRANCH_V(test_branch, float)
    
    
    
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
  
}

// Accumulate - Testing
//////////////////////////////
void MLClassification::accumulateTest(int target, const std::vector<float>& features) {

  accumulate(target, features, m_testing_data, m_testing_classes);
  
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


