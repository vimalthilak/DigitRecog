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

//root
#include <TH2F.h>
#include <TH3F.h>
#include <TH1F.h>
#include <TMultiGraph.h>
#include <TGraph.h>
#include <TGraphAsymmErrors.h>
#include <TVirtualFitter.h>
#include <TGraphErrors.h>
#include <TFitResultPtr.h>
#include <TFitResult.h>

#ifndef ROOT_Math_MinimizerOptions
#include "Math/MinimizerOptions.h"
#endif


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
void MLClassification::accumulate(int target, const std::vector<double>& features, cv::Mat * & data, cv::Mat * & classes) {
    
    
    if (!data) {
        
        init(features.size());
        
        data = new cv::Mat(0, m_nvar, cv::DataType<float>::type); //floats
        classes = new cv::Mat(0, 1, cv::DataType<int>::type); //integers
        
        data->reserve(10000);
        classes->reserve(10000);
        
    }
    
    if (m_nvar != features.size()) {LOG("Mismatch in features vector size.. got "<<
                                        features.size()<<", expecting: "<<m_nvar, logERROR); return;}
    
    
    std::vector<float> features_flt(features.begin(), features.end());
        
    cv::Mat input(features_flt, false); //copydata=false
    cv::Mat input_transpose = input.t();
    
    data->push_back(input_transpose); //transpose so we have one row with m_nvar columns
    
    classes->push_back(target);
}


// Accumulate - Training
//////////////////////////////
void MLClassification::accumulateTrain(int target, const std::vector<double>& features) {

  accumulate(target, features, m_training_data, m_training_classes);
  
  if (m_training_max.empty()) {
      LOG("Preparing max/min vectors..", logDEBUG);
      m_training_max.resize(m_nvar, -1.e99);
      m_training_min.resize(m_nvar, 1.e99);
  }
  
  for (std::size_t it_val = 0; it_val < m_nvar; ++it_val) {
      double val = features.at(it_val);
      if (val > m_training_max.at(it_val)) m_training_max[it_val] = val;
      if (val < m_training_min.at(it_val)) m_training_min[it_val] = val;
  }
  
}

// Accumulate - Testing
//////////////////////////////
void MLClassification::accumulateTest(int target, const std::vector<double>& features) {

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
    LOG(ss, logINFO);
    
    
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

    
    
    std::vector<char> th_successes;
    th_successes.resize(n_regions, true);
    
    std::vector<std::thread> threads;
    
    std::vector< std::unordered_map<unsigned int, std::pair<unsigned char,unsigned char> > > results;
    results.resize(n_regions);
                
    std::vector< std::unordered_map<unsigned int, std::map< unsigned char, float > > > results_prob;
    results_prob.resize(n_regions);

    //for progress bars
    for (unsigned int i = 0; i<n_regions; ++i) std::cerr<<std::endl;
    //atomic here is probably not necessary, we're only modifying the value once mutex is locked...
    std::atomic<int> console_offset(0);
    
    for (unsigned int i = 0; i<n_regions; ++i) {
    
       results[i].reserve(m_training_data->size().height/n_regions); //estimate
       results_prob[i].reserve(m_training_data->size().height/n_regions);
                    
       threads.push_back( std::thread(&MLClassification::train, this, std::ref(th_successes[i]), i, std::ref(console_offset), &RF_params,
                                     std::ref(cv_indices), std::ref(results[i]), std::ref(results_prob[i])) );
        
    }
    
    //LOG("Synchronizing threads...", logINFO);
    
    
    for (auto& th : threads) th.join();
    
    for (auto suc : th_successes) {
        if (!suc) { LOG("Prob with thread..", logERROR); return; }
    }
    
    LOG("...Done", logINFO);
    
    ////////////////////////////////
    // Analyzing results of cross-validation procedure.
    // Histograms/Graph written to current ROOT directory (i.e. last file opened, in this case from RootNtupleWriterTool)
    ////////////////////////////////
    
    /////////////////
    // Get handle on IncidentSvc
    /////////////////
    IncidentService * inc_svc = IncidentService::getInstance();
    if (!inc_svc) {LOG("Coulnd't get IncidentService", logERROR); return; }
    
    TH2F * res_correct = new TH2F("res_correct","res_correct", 110, 0, 1.1, 10, 0, 10 );
    TH2F * res_wrong = new TH2F("res_wrong","res_wrong", 110, 0, 1.1, 10, 0, 10 );
    
    ///
    //confusion matrix
    TH3F * results_matrix = new TH3F("results_matrix","results_matrix", 11,0,11, 11, 0, 11, 11, 0, 11);
    
    std::vector<TH1F*> reliability_vec;
    reliability_vec.reserve(10);
    
    std::vector<TH1F*> reliability_total_vec;
    reliability_total_vec.reserve(10);
    
    for (int i = 0; i < 10; ++i)  {
        std::ostringstream oss_;
        oss_<<"_"<<i;
        
        reliability_vec.push_back(new TH1F(("reliability"+oss_.str()).c_str(),("reliability"+oss_.str()).c_str(), 21, -0.025, 1.025) );
        reliability_vec.back()->Sumw2();
        reliability_vec.back()->SetDirectory(0); //do not auto-save histo
        
        reliability_total_vec.push_back(new TH1F(("reliability_total"+oss_.str()).c_str(),("reliability_total"+oss_.str()).c_str(), 21, -0.025, 1.025) );
        reliability_total_vec.back()->Sumw2();
        reliability_total_vec.back()->SetDirectory(0); //do not auto-save histo
    }
    
    float brier_score = 0.;
    for (std::size_t i_cv = 0; i_cv < results_prob.size(); ++i_cv) //loop over cv regions (1..n_regions)
        for (auto const & res_p : results_prob.at(i_cv) ) //loop over map of (sample, results)
        {
            
            unsigned char val_pred;
            unsigned char val_truth;
            try {
                val_pred =  results.at(i_cv).at(res_p.first).second;
                val_truth = results.at(i_cv).at(res_p.first).first;  }
            catch (const std::out_of_range& oor) { LOG("Out of Range error: " << oor.what(), logWARNING); break;}
            
            bool correct = (val_pred == val_truth);
            
            TH2F * hist = (correct) ? res_correct : res_wrong ;
            hist->Fill( res_p.second.at(val_pred),  static_cast<float>(val_pred) + 0.5 );
            
            /////////
            //confusion matrix
            results_matrix->Fill(static_cast<float>(val_truth)+0.5, static_cast<float>(val_pred)+0.5, 0.5);
            //std::pair<unsigned char, float>
            const auto & largest_prob = *map_max_element(res_p.second);
            //alternative:
            int prediction =  static_cast<int>(largest_prob.first);
            results_matrix->Fill(static_cast<float>(val_truth)+0.5, static_cast<float>(prediction)+0.5, 1.5);
            ///////////////
            
            
            //////////////////
            // Start filling TTree with results
            //////////////////
            inc_svc->fireIncident(Incident("BeginEvent"));
            
            int target = static_cast<int>(val_truth);
            int pred = static_cast<int>(val_pred);
            
            PUSHBACK(target, target);
            PUSHBACK(prediction, pred);
            
            
            //brier score:
            for (auto const & res_p_individual : res_p.second)
                brier_score += ( val_truth == res_p_individual.first ) ? (res_p_individual.second - 1.)*(res_p_individual.second - 1.)
                : (res_p_individual.second )*(res_p_individual.second );
            
            for (int i = 0; i < 10; ++i) {
                //reliability
                if ((int)val_truth == i) reliability_vec[i]->Fill(res_p.second.at(i));
                reliability_total_vec[i]->Fill(res_p.second.at(i));
                
                PUSHBACK(prediction_prob, res_p.second.at(i));
                
            }
            
            // signal end of event --> flushes data to TTree
            inc_svc->fireIncident(Incident("EndEvent"));
            
            
        }
    
    LOG("Brier score: "<<brier_score/m_training_data->size().height, logINFO);
    
    /////////////
    // confusion matrix post-process
    ////////////
    
    //compute total # of hits per row/column
    for (int z_bin = 1; z_bin <= results_matrix->GetNbinsZ(); ++z_bin ) {
        
        for (int i_bin = 1; i_bin< results_matrix->GetNbinsX(); ++i_bin ) {
            
            int tot = results_matrix->Integral(i_bin,i_bin, 0,10, z_bin, z_bin);
            results_matrix->SetBinContent(i_bin, results_matrix->GetNbinsY(),z_bin, tot);
            
        }
        
        
        for (int i_bin = 1; i_bin< results_matrix->GetNbinsY(); ++i_bin ) {
            
            int tot = results_matrix->Integral(0,10, i_bin,i_bin, z_bin, z_bin);
            results_matrix->SetBinContent(results_matrix->GetNbinsX(), i_bin, z_bin, tot);
        }
        
    }
    ///////////
    
    std::vector<TGraphAsymmErrors*> reliability_graph_vec;
    reliability_graph_vec.reserve(10);
    
    ROOT::Math::MinimizerOptions::SetDefaultMaxFunctionCalls(50000);
    
    for (int i = 0; i < 10; ++i) {
        TGraphAsymmErrors *reliability_graph = new TGraphAsymmErrors();
        reliability_graph_vec.push_back(reliability_graph);
        
        reliability_graph->Divide(reliability_vec[i],reliability_total_vec[i],"cl=0.683 b(1,1) mode e0");
        
        //fitting
        TF1 * f1 = new TF1("f1", "1/TMath::Power((1 + [0]*exp(-1*[1]*(x-[2]))),1./[3])", 0.01, 1.01); //skip first bin...giving us trouble..
        f1->SetParameters(1,10,0.5,1);
        
        TF1 * f2 = new TF1("f2", "1/(1 + exp(-1*[0]*(TMath::Power(x,[2])-[1])))", 0.01, 1.01);
        f2->SetLineColor(kBlue);
        f2->SetParameters(10,0.5,0.5);
        
        TF1 * f3 = new TF1("f3", "1-exp(-1.*[0]*(TMath::Power(x,[1])))", 0.01, 1.01);
        f3->SetLineColor(kGreen);
        f3->SetParameters(10,2);
        
        //(TVirtualFitter::GetFitter())->Clear();
        //f1->SetParLimits(3,0, 10);
        TFitResultPtr fit = reliability_graph->Fit(f1, "ESQR+"); //TFitResultPtr fit =  reliability_graph->Fit(f1, "EMSQ"); // fit->Chi2()/fit->Ndf()
        float fit_chi2_ndf_1 = fit->Chi2()/fit->Ndf();
        
        
        //if (fit_res) std::cout<<"fit result non zero for "<<i<<": "<<fit_res<<std::endl;
        
        TGraphErrors *grint = new TGraphErrors(reliability_graph->GetN()-1);
        for (int ii_n = 0; ii_n < grint->GetN(); ++ii_n) grint->SetPoint(ii_n, reliability_graph->GetX()[ii_n+1],0);//skip first bin
        (TVirtualFitter::GetFitter())->GetConfidenceIntervals(grint, 0.683);
        grint->SetLineColor(kRed);
        
        
        //flush to file
        std::ostringstream oss_;
        oss_<<"_"<<i;
        grint->Write(("reliability_graph1_68cl"+oss_.str()).c_str());
        delete grint;
        
        //f2////
        (TVirtualFitter::GetFitter())->Clear();
        fit = reliability_graph->Fit(f2, "ESQR+"); //TFitResultPtr fit =  reliability_graph->Fit(f1, "EMSQ"); // fit->Chi2()/fit->Ndf()
        float fit_chi2_ndf_2 = fit->Chi2()/fit->Ndf();
        
        grint = new TGraphErrors(reliability_graph->GetN()-1);
        for (int ii_n = 0; ii_n < grint->GetN(); ++ii_n) grint->SetPoint(ii_n, reliability_graph->GetX()[ii_n+1],0); //skip first bin
        (TVirtualFitter::GetFitter())->GetConfidenceIntervals(grint, 0.683);
        grint->SetLineColor(kBlue);
        grint->Write(("reliability_graph2_68cl"+oss_.str()).c_str());
        delete grint;
        ////
        
        //f3////
        (TVirtualFitter::GetFitter())->Clear();
        fit = reliability_graph->Fit(f3, "ESQR+"); //TFitResultPtr fit =  reliability_graph->Fit(f1, "EMSQ"); // fit->Chi2()/fit->Ndf()
        float fit_chi2_ndf_3 = fit->Chi2()/fit->Ndf();
        
        grint = new TGraphErrors(reliability_graph->GetN()-1);
        for (int ii_n = 0; ii_n < grint->GetN(); ++ii_n) grint->SetPoint(ii_n, reliability_graph->GetX()[ii_n+1],0); //skip first bin
        (TVirtualFitter::GetFitter())->GetConfidenceIntervals(grint, 0.683);
        grint->SetLineColor(kGreen);
        grint->Write(("reliability_graph3_68cl"+oss_.str()).c_str());
        delete grint;
        ////
        
        
        LOG("f1: "<<fit_chi2_ndf_1<<"  f2: "<<fit_chi2_ndf_2<<"  f3: "<<fit_chi2_ndf_3, logINFO);
        
        
        //flush to file
        reliability_graph->Write(("reliability_graph"+oss_.str()).c_str());
        
        
        (TVirtualFitter::GetFitter())->Clear();
        
        
        float brier_reliability_score = 0.;
        for (int i_bin = 1; i_bin <= reliability_total_vec[i]->GetNbinsX(); ++ i_bin) {
            
            float o_k = (reliability_total_vec[i]->GetBinContent(i_bin)>0) ?
            reliability_vec[i]->GetBinContent(i_bin)/reliability_total_vec[i]->GetBinContent(i_bin)
            : 0.;
            float f_k = reliability_vec[i]->GetBinCenter(i_bin);
            brier_reliability_score += reliability_total_vec[i]->GetBinContent(i_bin) * (f_k-o_k) *(f_k-o_k);
            
        }
        
        LOG("Brier reliability score for "<<i<<": "<<brier_reliability_score/m_training_data->size().height, logINFO);
        
        //clean up
        //delete reliability_graph;
        delete f1;
        delete f2;
        delete f3;
        
        delete reliability_vec[i];
        delete reliability_total_vec[i];
    }
    
    reliability_vec.clear();
    reliability_total_vec.clear();
    

    
    
    //calibration - sanity plots
    
    TH1F * normalization = new TH1F("normalization", "normalization", 100, 0, 2);
    
    TH2F * max_prob = new TH2F("max_prob", "max_prob", 10, 0, 10, 10, 0, 10);
    
    for (std::size_t i_cv = 0; i_cv < results_prob.size(); ++i_cv) //loop over cv regions (1..n_regions)
        for (auto const & res_p : results_prob.at(i_cv) ) //loop over map of (sample, results)
        {
            
            float norm = 0;
            std::map<int, float> new_prob;
            for (auto const & res_p_map : res_p.second ) {
                
                std::size_t num = (int)res_p_map.first;
                float score = res_p_map.second;
                
                float prob = reliability_graph_vec.at(num)->GetFunction("f3")->Eval(score);
                new_prob[num] = prob;
                norm += prob;
                
            }
            
            for (auto & n : new_prob)
                n.second /= norm;
            
            
            int largest_prob_bef = (int)(*map_max_element(res_p.second)).first;
            int largest_prob_aft = (*map_max_element(new_prob)).first;
            
            max_prob->Fill(largest_prob_bef+0.5, largest_prob_aft+0.5);
            
            normalization->Fill(norm);
            
            
        }
    
    //clean up
    for (auto & g : reliability_graph_vec)
        delete g;
    
    reliability_graph_vec.clear();
    
    
        
    


}


void MLClassification::train(char & success, const int i,std::atomic<int>& console_offset,  const CvRTParams* RF_params,
                      const std::vector< std::vector<int> >& cv_indices,
                      std::unordered_map<unsigned int, std::pair<unsigned char,unsigned char> > & results,
                      std::unordered_map<unsigned int, std::map< unsigned char, float > > & results_prob) {
    

#define CATCH(str)  catch (const std::exception& ex) { \
                      std::lock_guard<std::mutex> lock(m_log_mtx); \
                      LOG("Caught Exception in " #str " : " <<ex.what(), logWARNING); \
                      success = false;\
                      return;\
                    }\
                    catch(...) {\
                      std::lock_guard<std::mutex> lock(m_log_mtx);\
                      LOG("Caught unknown Exception in " #str, logWARNING);\
                      success = false;\
                      return;\
                    }     
    
   const cv::Mat& training_data = *m_training_data;
   const cv::Mat& training_classes = *m_training_classes;
   const cv::Mat& var_type = *m_var_type;
   
   unsigned int line_num = training_data.size().height;
   
   cv::Mat sample_idx(line_num, 1, cv::DataType<unsigned char>::type, cv::Scalar(1)); //ones everywhere
   for ( auto cv : cv_indices.at(i) ) sample_idx.at<unsigned char>(cv) = 0;
   
   // training
   MyCvRTrees RF(cv_indices.size()-i, &console_offset);
   try {
      RF.train(training_data, CV_ROW_SAMPLE, training_classes, cv::Mat(), sample_idx, var_type, cv::Mat(), *RF_params);
   }
   CATCH(MLClassification::train)
   
   if(true) {
        std::lock_guard<std::mutex> lock(m_log_mtx);
       
        //also need to lock mutex from progress bars..
       
        std::lock_guard<std::mutex> lock_bar(MyCvRTrees::m_log_mtx);
       
        LOG("Done training, thread "<<i, logINFO);
       
        //increase offset by 1 :
        console_offset++;
   }
   
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
           success = false;
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


