#include "IPreProcessorTools.h"

//helper macro for 2D indices access to a 1D vector
#define AT2D(vec, col, x, y) vec.at( ((col-1)-(y))*col + (x)) 

class GaussianBlurTool : virtual public IPreProcessorTools {
    
    
public:
    GaussianBlurTool() {};
    ~GaussianBlurTool() {};
    
    void execute(std::vector<float> &);
    
    template<std::size_t N_K>
    static void conv_2d_square(const float kernel[][N_K], const std::vector<float>&,
                                                      std::vector<float>&);
    
private:
    
    //more blur
    const float gaussian_filter[3][3] = {
        
        {1./16, 2./16, 1./16},
        {2./16, 4./16, 2./16},
        {1./16, 2./16, 1./16}
    };
    
    //less blur
    const float gaussian_filter_2[3][3] = {
        
        {1./23.28,    2.82/23.28, 1./23.28},
        {2.82/23.28,  8./23.28,   2.82/23.28},
        {1./23.28,    2.82/23.28, 1./23.28}
    };
    
    
};
