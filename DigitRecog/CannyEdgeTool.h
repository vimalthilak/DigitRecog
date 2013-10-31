#include "IPreProcessorTools.h"
#include <utility>      // std::pair
#include <set>

class CannyEdgeTool : virtual public IPreProcessorTools {
    
    
public:
    CannyEdgeTool() {};
    ~CannyEdgeTool() {};
    
    void execute(std::vector<float> &);
    
    float GetLowHystThreshold2() { return m_hysteresis_thresholds[0]; }
    float GetHighHystThreshold2() { return m_hysteresis_thresholds[1]; }
    
    void SetLowHystThreshold2(float t) { m_hysteresis_thresholds[0] = t; }
    void SetHighHystThreshold2(float t) { m_hysteresis_thresholds[1] = t; }
    
    
protected:

    void connect(int, int, const std::set<std::pair<int, int> >&, const std::vector<int>&, std::set< std::pair<int, int> > &);
    


    float m_hysteresis_thresholds[2] {5000., 10000.} ;
    
    const float sobel_gy[3][3] = {
        
        {1./4, 0, -1./4},   //[0][0], [0][1], [0][2]
        {2./4, 0, -2./4},   //[1][0], [1][1], [1][2]
        {1./4, 0, -1./4}    //[2][0], [2][1], [2][2]
    };
    
    const float sobel_gx[3][3] = {
        
        {1./4, 2./4,    1./4},
        {0,    0,       0},
        {-1./4, -2./4, -1./4}
    };
    
};
