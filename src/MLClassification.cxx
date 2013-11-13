#include "DigitRecog/MLClassification.h"
#include "UtilityToolsInterfaces/IRootNtupleWriterTool.h"
#include "UtilityToolsInterfaces/SingleObjectHolder.h"
#include "UtilityToolsInterfaces/VectorObjectHolder.h"
#include "Services/IncidentService.h"
#include "Services/Incident.h"

//opencv huge include file
#include "opencv/ml.h"
#include "rtrees.hpp"

//c++11
#include <random>
#include <thread>


#include <cmath>
#include <set>


//helper macros
#define REGBRANCH_S(name, type, initVal) m_ntuple_helper->registerBranch(#name, new SingleObjectHolder< type >(initVal));
#define REGBRANCH_V(name, type)          m_ntuple_helper->registerBranch(#name, new VectorObjectHolder< type >());
#define PUSHBACK(name, obj)      pushBack(#name, &(obj), m_ntuple_helper );


// Constructor
///////////////////////////
MLClassification::MLClassification(const std::string& n):
        Messaging(n), m_training_data(0), m_training_classes(0), m_var_type(0),
                      m_testing_data(0), m_testing_classes(0),
        m_ntuple_helper(0), m_nvar(0) {
    
    Register();
    
}

///////////////////////////
MLClassification::MLClassification(const std::string& n, TLogLevel m_lvl):
         Messaging(n, m_lvl), m_training_data(0), m_training_classes(0), m_var_type(0),
                      m_testing_data(0), m_testing_classes(0),
         m_ntuple_helper(0), m_nvar(0) {
    
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
    if (!m_ntuple_helper) return 0;
    
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
void MLClassification::accumulate(int target, const std::vector<float>& features, cv::Mat * & data, cv::Mat * & classes) {
    
    
    if (!data) {
        
        init(features.size());
        
        data = new cv::Mat(0, m_nvar, cv::DataType<float>::type); //floats
        classes = new cv::Mat(0, 1, cv::DataType<int>::type); //integers
        
        data->reserve(10000);
        classes->reserve(10000);
        
    }
    
    if (m_nvar != features.size()) {LOG("Mismatch in features vector size.. got "<<
                                        features.size()<<", expecting: "<<m_nvar, logERROR); return;}
    
    
        
    cv::Mat input(features, false); //copydata=false
    cv::Mat input_transpose = input.t();
    
    data->push_back(input_transpose); //transpose so we have one row with m_nvar columns
    
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
void MLClassification::performCrossValidationTraining(unsigned int n_regions,
                                                      int max_depth, int min_sample, int num_var,
                                                      int num_trees) { //max_tress=500 (default) 

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
    
    std::string ss = "sizes of cross_validation regions: ";
    for ( std::size_t i = 0; i < cv_indices.size(); ++i) {
       std::ostringstream oss;
       oss<<cv_indices[i].size()<< ", ";
       ss += oss.str();
    }
    LOG(ss, logDEBUG);
    
    
    /////////// RF param
    CvRTParams RF_params( max_depth,        // max depth
                         min_sample,        // min sample count
                                  0,        // regression accuracy: N/A here
                              false,        // compute surrogate split, no missing data
                                 15,        // max number of categories (use sub-optimal algorithm for larger numbers)
                                  0,        // the array of priors
                              false,        // calculate variable importance
                            num_var,        // number of variables randomly selected at node and used to find the best split(s). 0 -> sqrt(NVAR)
                          num_trees,        // max number of trees in the forest
                            0.0001f,        // forrest accuracy
                    CV_TERMCRIT_ITER        // | CV_TERMCRIT_EPS  // termination cirteria
                                    );

    
    
    
    /////test
    /////////////////
    // Get handle on IncidentSvc
    IncidentService * inc_svc = IncidentService::getInstance();
    if (!inc_svc) {LOG("Coulnd't get IncidentService", logERROR); return; }

    
    for (int j = 0; j< 5; ++j) {
    
        inc_svc->fireIncident(Incident("BeginEvent"));
        
        PUSHBACK(target, j);
        
        inc_svc->fireIncident(Incident("EndEvent"));
    }
    
   




}



void MLClassification::train(const int i, const CvRTParams* RF_params,
                      const std::vector< std::vector<int> >& cv_indices,
                      std::unordered_map<unsigned int, std::pair<unsigned char,unsigned char> > & results,
                      std::unordered_map<unsigned int, std::map< unsigned char, float > > & results_prob) {
    

#define CATCH(str)  catch (const std::exception& ex) { \
                      std::lock_guard<std::mutex> lock(m_log_mtx); \
                      LOG("Caught Exception in " #str " : " <<ex.what(), logWARNING); \
                      return;\
                    }\
                    catch(...) {\
                      std::lock_guard<std::mutex> lock(m_log_mtx);\
                      LOG("Caught unknown Exception in " #str, logWARNING);\
                      return;\
                    }
    
   const cv::Mat& training_data = *m_training_data;
   const cv::Mat& training_classes = *m_training_classes;
   const cv::Mat& var_type = *m_var_type;
   
   unsigned int line_num = training_data.size().height;
   
   cv::Mat sample_idx(line_num, 1, cv::DataType<unsigned char>::type, cv::Scalar(1)); //ones everywhere
   for ( auto cv : cv_indices.at(i) ) sample_idx.at<unsigned char>(cv) = 0;
   
   // training
   MyCvRTrees RF;
   try {
      RF.train(training_data, CV_ROW_SAMPLE, training_classes, cv::Mat(), sample_idx, var_type, cv::Mat(), *RF_params);
   }
   CATCH(MLClassification::train)
   
   // retrieving results
   cv::Mat test_sample;
   
   for (unsigned int tsample = 0; tsample < line_num; ++tsample) //line_num
   {
        if ( sample_idx.at<unsigned char>(tsample) == 1 ) continue;
        
        
        // extract a row from the testing matrix
        
        test_sample = training_data.row(tsample);
        
        // run random forest prediction
        
        double result = -1.;
        try {
           result = RF.predict(test_sample);
        }
        CATCH(RF.predict)
       
        
        // if the prediction and the (true) testing classification are the same
        // (N.B. openCV uses a floating point decision tree implementation!)
        
        int res_int = (int)(result+0.1);
        
        results[tsample] = std::make_pair(static_cast<unsigned char>(training_classes.at<int>(tsample)), static_cast<unsigned char>(res_int));
        
        //probabilities
        std::vector< std::pair<float, float> > results_probabilities;
        bool is_good = false;
        try {
           is_good = RF.predict_prob_multiclass( test_sample, cv::Mat(), results_probabilities);
        }
        CATCH(RF.predict_prob_multiclass)
       
        if ( !is_good) {
           std::lock_guard<std::mutex> lock(m_log_mtx);
           LOG("RF.predict_prob_multiclass failed..", logWARNING)
           break;
        }
        
        auto& res_prob = results_prob[tsample];
        for (auto const & res : results_probabilities) {
            unsigned char value = static_cast<unsigned char>(res.first+0.1);
            res_prob[value] = res.second;
        
        }
   }
    
#undef CATCH
    
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


