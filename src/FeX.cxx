#include "DigitRecog/FeX.h"
#include "UtilityToolsInterfaces/IRootNtupleWriterTool.h"
#include "UtilityToolsInterfaces/SingleObjectHolder.h"
#include "UtilityToolsInterfaces/VectorObjectHolder.h"
#include "Services/IncidentService.h"
#include "DigitRecog/IPreProcessorTools.h"
#include <cmath>
#include <utility>      // std::pair
#include <set>

//helper macros
#define REGBRANCH_S(name, type, initVal) m_ntuple_helper->registerBranch(#name, new SingleObjectHolder< type >(initVal));
#define REGBRANCH_V(name, type)          m_ntuple_helper->registerBranch(#name, new VectorObjectHolder< type >());
#define PUSHBACK(name, obj)              pushBack(#name, &(obj), m_ntuple_helper );

#define REGBRANCH_S_META(name, type, initVal) m_ntuple_helper_metadata->registerBranch(#name, new SingleObjectHolder< type >(initVal));
#define REGBRANCH_V_META(name, type)          m_ntuple_helper_metadata->registerBranch(#name, new VectorObjectHolder< type >());
#define PUSHBACK_META(name, obj)              pushBack_meta<std::string>(#name, obj, m_ntuple_helper_metadata);


// Constructor
///////////////////////////
FeX::FeX(const std::string& n, IRootNtupleWriterTool* r, IRootNtupleWriterTool* r_meta):
        Messaging(n), m_first(false), m_ntuple_helper(r), m_ntuple_helper_metadata(r_meta) {
    
    Register();
    
}

///////////////////////////
FeX::FeX(const std::string& n, TLogLevel m_lvl, IRootNtupleWriterTool* r, IRootNtupleWriterTool* r_meta):
         Messaging(n, m_lvl), m_first(false), m_ntuple_helper(r), m_ntuple_helper_metadata(r_meta) {
    
    Register();
    
}



void FeX::Register() {

    // Get handle on IncidentSvc
    IncidentService * inc_svc = IncidentService::getInstance();
    if (!inc_svc) {LOG("Coulnd't get IncidentService", logERROR); return; }
    
    inc_svc->addListener(this, "BeginRun");

}

// Initialization
////////////////////////////
int FeX::initialize()
{
    
    LOG("Initialization..",logDEBUG);
    
    REGBRANCH_S(target, int, -1)
    
    REGBRANCH_V(features, float)

    //REGBRANCH_V(test_branch, float)
    
    REGBRANCH_V_META(features_name, std::string);
    
    return 1;
    
}

//
////////////////////////////
void FeX::addPreProcessorTool(IPreProcessorTools* t) {
    
    m_pre_tools.push_back(t);
    
}


// Execute
//////////////////////////////
int FeX::execute(int target, std::vector<float>& img) {
    
#define AT2D(vec, col, x, y) vec.at( ((col-1)-(y))*col + (x))
#define AT2D_(vec, x, y) AT2D(vec, dim, x, y)
    
    
    //LOG("Error Test!!",logERROR);
    //LOG("WARNING Test!!",logWARNING);
    
    LOG("In execute()", logVERBOSE);
    
    PUSHBACK(target, target)
    
    //assume we're giving a square matrix
    std::size_t dim = (unsigned int)(std::sqrt(img.size()) + 0.01);
    
    //--- preprocessing
    
    //gaussian blur
    
    //canny
    //sobel gx, gy -> angle/direction
    //non-max suppression
    //tracing
    
    LOG("Executing PreProcessorTools...", logVERBOSE);
    
    //modify the image in-place
    for (auto tool : m_pre_tools)
        tool->execute(img);
    
    //--- features
    
    //cm
    //inertia moments (dr_max, dr_min axis)
    //symmetry wrt dr_max/dr_min
    //five number summary per quadrant
    
    
    // Center of Mass
    // as a first pass, store non-zero values for faster future accesses
    // assuming a binary image ...
    
    std::set<std::pair<int, int> > results;
    
    float x_cm = 0.;
    float y_cm = 0.;
    for (std::size_t i_x  = 0; i_x < dim; ++i_x) {
        for (std::size_t i_y  = 0; i_y < dim; ++i_y) {
            
            int pixel_value = static_cast<int>(AT2D_(img, i_x, i_y) + 0.01);
            if ( pixel_value == 0 ) continue;
            
            results.emplace(i_x, i_y);
            
            x_cm += pixel_value*i_x;
            x_cm += pixel_value*0.5; //center
            
            y_cm += pixel_value*i_y;
            y_cm += pixel_value*0.5; //center
            
            
        }
    }
    
    x_cm /= results.size();
    y_cm /= results.size();
    
    PUSHBACK(features, x_cm)
    PUSHBACK(features, y_cm)
    
    if (!m_first) {
       PUSHBACK_META(features_name, "center_of_mass_x")
       PUSHBACK_META(features_name, "center_of_mass_y")
    }
    
    //minimize second moment (inertia)
    float a = 0;
    float b = 0;
    float c = 0;
    for ( auto const & it : results ) {
        
        float x_prime = it.first+0.5 -  x_cm;
        float y_prime = it.second+0.5 - y_cm;
        
        a += x_prime*x_prime;
        b += 2*x_prime*y_prime;
        c += y_prime*y_prime;
        
    }
    
    //(b) x^2 + 2(a-c) x -b = 0, x= tan(theta)
    //(2*(c-a) + std::sqrt(4*(a-c)*(a-c) - 4*b*(-b)) ) /(2*b)
    
    float tan_positive = ((c-a) + std::sqrt((a-c)*(a-c) + b*b) );
    float tan_negative = ((c-a) - std::sqrt((a-c)*(a-c) + b*b) );
    
    float theta_pos = (std::fabs(b)> 1.e-6) ? std::atan(tan_positive/b) : (tan_positive/std::fabs(tan_positive)) * std::acos(0.); //+/- pi/2
    float theta_neg = (std::fabs(b)> 1.e-6) ? std::atan(tan_negative/b) : 0.; //(tan_negative/std::fabs(tan_negative)) * std::acos(0.);
    
    float E_pos = a*std::sin(theta_pos)*std::sin(theta_pos) -b*std::sin(theta_pos)*std::cos(theta_pos)+c*std::cos(theta_pos)*std::cos(theta_pos);
    float E_neg = a*std::sin(theta_neg)*std::sin(theta_neg) -b*std::sin(theta_neg)*std::cos(theta_neg)+c*std::cos(theta_neg)*std::cos(theta_neg);
    
    
    /////// interia moments /////////////
    
    PUSHBACK(features, E_pos)
    PUSHBACK(features, E_neg)
    PUSHBACK(features, theta_pos)
    PUSHBACK(features, theta_neg)
    
    if (!m_first) {
       PUSHBACK_META(features_name, "I_pos")
       PUSHBACK_META(features_name, "I_neg")
       PUSHBACK_META(features_name, "I_theta_pos")
       PUSHBACK_META(features_name, "I_theta_neg")
    }
    
    //float tan_theta_min = (E_pos < E_neg) ? tan_positive/b : tan_negative/b;
    //float tan_theta_max = (E_pos > E_neg) ? tan_positive/b : tan_negative/b;
        
    float theta_min =  (E_pos < E_neg) ? theta_pos : theta_neg;
    float theta_max =  (E_pos > E_neg) ? theta_pos : theta_neg;
    
    //E_max not really the max..just line orthogonal to minimized axis
    // can't maximize.. just take axis at infinity, initeria = infinity, ...
    
    ///
    /// partition image in four quadrants according to axes defined by theta_min/max
    
    std::vector<float> r_min[4];
    std::vector<float> r_max[4];
    std::vector<float> ang_wrtmin[4];
    
    
    // in order to define some symmetry observable w.r.t. axes
    std::vector< std::pair<float, float> > above_max;
    std::vector< std::pair<float, float> > below_max;
    
    std::vector< std::pair<float, float> > above_min;
    std::vector< std::pair<float, float> > below_min;
    
    for ( auto const & it : results ) {
        
        //change of coordinates
        float x = it.first+0.5 - x_cm;
        float y = it.second+0.5 - y_cm;
        
        float dr_min = (x)*std::sin(theta_min) - (y)*std::cos(theta_min);
        float dr_max = (x)*std::sin(theta_max) - (y)*std::cos(theta_max);
        
        int i = -1;
        if ( dr_min > 0 && dr_max > 0) i = (theta_pos< 0.) ? 2 : 3;
        else if ( dr_min > 0 && dr_max <= 0) i = (theta_pos< 0.) ? 1 : 0;
        else if ( dr_min <= 0 && dr_max > 0) i = (theta_pos< 0.) ? 3 : 2;
        else if ( dr_min <= 0 && dr_max <= 0) i = (theta_pos< 0.) ? 0 : 1;
        else { LOG("What? quadrant . "<<a<<", "<<b<< "  "<<c<<"   theta: "<<theta_min<<", "<<theta_max<<" ... "<< tan_negative, logERROR); }
        
        
        r_min[i].push_back(std::fabs(dr_min));
        r_max[i].push_back(std::fabs(dr_max));
        ang_wrtmin[i].push_back(std::atan( std::fabs( dr_min / dr_max )) );
        
        if (i ==0 || i==2) above_max.emplace_back(dr_min, dr_max);
        else if (i ==1 || i==3) below_max.emplace_back(dr_min, dr_max);
        
        if (i ==0 || i==1) above_min.emplace_back(dr_min, dr_max);
        else if (i ==2 || i==3) below_min.emplace_back(dr_min, dr_max);
        
    }
    
    //symmetry w.r.t dr_max/dr_min
    for (int i_axis = 0; i_axis<2; ++i_axis) {
        
        const auto & above = (i_axis==0) ? above_max :  above_min;
        const auto & below = (i_axis==0) ? below_max :  below_min;
        
        for (int i_set = 0; i_set<2; ++i_set) {
            
            const auto & set_1 = (i_set==0) ? above :  below;
            const auto & set_2 = (i_set==0) ? below :  above;
            
            float symm_max = 0.;
            for (const auto & it_1 : set_1 ) {
                float x_1 = (i_axis == 0) ? it_1.first       :  it_1.first *-1.;    //invert x coordinates (reflect points on the other side)
                float y_1 = (i_axis == 0) ? it_1.second *-1. :  it_1.second; //invert y coordinates
                
                double min_dist = 1e99;
                //find closest point in other set (i.e. other side of the dr_max axis):
                for (const auto & it_2 : set_2) {
                    float x_2 = it_2.first;
                    float y_2 = it_2.second;
                    float dist_squared = (x_2-x_1)*(x_2-x_1) + (y_2-y_1)*(y_2-y_1);
                    
                    if (dist_squared<min_dist) min_dist = dist_squared;
                }
                
                symm_max += std::sqrt(min_dist);
            }
            
            /////////
            symm_max /= set_1.size();
            PUSHBACK(features, symm_max);
            if (!m_first) {
                std::ostringstream oss;
                oss<<"Symm_"<< ((i_axis == 0) ? "max" : "min") << ((i_set == 0) ? "_above" : "_below");
                PUSHBACK_META(features_name, oss.str())
            }
            ///////
        }
    }

    //five number summary
    float r_min_stats[4][5];
    float r_max_stats[4][5];
    float ang_wrtmin_stats[4][5];
    
    for (int i =0; i < 4; ++i) {
        
        if ( r_min[i].size() < 1 ) continue;
        
        std::sort( r_min[i].begin(), r_min[i].end() );
        std::sort( r_max[i].begin(), r_max[i].end() );
        std::sort( ang_wrtmin[i].begin(), ang_wrtmin[i].end() );
        
        //min
        r_min_stats[i][0] = r_min[i].front();
        r_max_stats[i][0] = r_max[i].front();
        ang_wrtmin_stats[i][0] = ang_wrtmin[i].front();
        
        
        //first quartile
        r_min_stats[i][1] = CalcMedian<float>(r_min[i].begin(), r_min[i].begin() + r_min[i].size()/2 );
        r_max_stats[i][1] = CalcMedian<float>(r_max[i].begin(), r_max[i].begin() + r_max[i].size()/2 );
        ang_wrtmin_stats[i][1] = CalcMedian<float>(ang_wrtmin[i].begin(), ang_wrtmin[i].begin() + ang_wrtmin[i].size()/2 );
        
        //median
        r_min_stats[i][2] = CalcMedian<float>(r_min[i].begin(), r_min[i].end());
        r_max_stats[i][2] = CalcMedian<float>(r_max[i].begin(), r_max[i].end());
        ang_wrtmin_stats[i][2] = CalcMedian<float>(ang_wrtmin[i].begin(), ang_wrtmin[i].end());
        
        
        //third quartile
        r_min_stats[i][3] = CalcMedian<float>(r_min[i].end() - r_min[i].size()/2, r_min[i].end() );
        r_max_stats[i][3] = CalcMedian<float>(r_max[i].end() - r_max[i].size()/2, r_max[i].end() );
        ang_wrtmin_stats[i][3] = CalcMedian<float>(ang_wrtmin[i].end() - ang_wrtmin[i].size()/2, ang_wrtmin[i].end() );
        
        //max
        r_min_stats[i][4] = r_min[i].back();
        r_max_stats[i][4] = r_max[i].back();
        ang_wrtmin_stats[i][4] = ang_wrtmin[i].back();
        
        //////////// five number summary ///////////////
        for (int j = 0; j < 5; ++j) {
            //r_min
            PUSHBACK(features, r_min_stats[i][j]);
            
            //r_max
            PUSHBACK(features, r_max_stats[i][j]);
            
            //ang_wrtmin
            PUSHBACK(features, ang_wrtmin_stats[i][j]);
            
            if (!m_first) {
                
                std::string n;
                switch(j) {
                   case 0:
                       n = "min";
                       break;
                   case 1:
                       n = "first_quartile";
                       break;
                   case 2:
                       n = "median";
                       break;
                   case 3:
                       n = "third_quartile";
                       break;
                   case 4:
                       n = "max";
                       break;
                }
                
                std::ostringstream oss;
                oss<<"quadrant["<<i<<"]_r-wrtmin_"<<n;
                PUSHBACK_META(features_name, oss.str());
                
                oss.str("");
                oss<<"quadrant["<<i<<"]_r-wrtmax_"<<n;
                PUSHBACK_META(features_name, oss.str());
                
                oss.str("");
                oss<<"quadrant["<<i<<"]_ang-wrtmin_"<<n;
                PUSHBACK_META(features_name, oss.str());
            
            } //if !m_first
        }
        
        ////////////////////////////////////////////////
        
        LOG("quadrant "<<i, logVERBOSE);
        LOG("  r_min      "<<r_min_stats[i][0]<<", "<<r_min_stats[i][1]<<", "<<r_min_stats[i][2]<<", "<<r_min_stats[i][3]<<", "<<r_min_stats[i][4], logVERBOSE);
        LOG("  r_max      "<<r_max_stats[i][0]<<", "<<r_max_stats[i][1]<<", "<<r_max_stats[i][2]<<", "<<r_max_stats[i][3]<<", "<<r_max_stats[i][4], logVERBOSE);
        LOG("  ang_wrtmin "<<ang_wrtmin_stats[i][0]<<", "<<ang_wrtmin_stats[i][1]<<", "<<ang_wrtmin_stats[i][2]<<", "<<ang_wrtmin_stats[i][3]<<", "<<ang_wrtmin_stats[i][4], logVERBOSE);
        
    }
    
#undef AT2D_
#undef AT2D

    
    m_first = true;
    
    return 1;
}

//helper method
template <class T>
float FeX::CalcMedian( typename std::vector<T>::const_iterator scores_begin, typename std::vector<T>::const_iterator scores_end )
{
    float median;
    std::size_t size = std::distance(scores_begin, scores_end);
    
    if (size == 0) return *(scores_begin);
    
    if (size  % 2 == 0)
    {
        median =  (*(scores_begin + (size/2-1)) + *(scores_begin + size/2) )/2; //(scores[size / 2 - 1] + scores[size / 2]) / 2;
    }
    else
    {
        median =  *(scores_begin + size/2); //scores[size / 2];
    }
    
    return median;
}


////////////////////////////////////////////////////////////////////////////////
// Handle incidents
////////////////////////////////////////////////////////////////////////////////
void FeX::handle(const Incident& inc)
{
    
    if (inc.svcType() == "BeginRun") {
        
        if ( !(this->initialize())) { LOG("Couldn't initialize properly..", logERROR); }
        
    } //else if
    
    
    
} //void

template <class T>
inline void FeX::pushBack(const std::string & branch_name, const T* obj, IRootNtupleWriterTool * m_ntuple) {
    m_ntuple->pushBack(branch_name, boost::any(obj));
}

template <class T>
inline void FeX::pushBack_meta(const std::string & branch_name, const T& obj, IRootNtupleWriterTool * m_ntuple) {
    m_ntuple->pushBack(branch_name, boost::any(&obj));
}


