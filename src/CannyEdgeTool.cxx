#include "DigitRecog/CannyEdgeTool.h"
#include "DigitRecog/GaussianBlurTool.h"
#include <cmath>
 

//#define AT2D_(vec, x, y) AT2D(vec, cols, x, y)

void CannyEdgeTool::execute(std::vector<float> & img) {
    
#define AT2D_(vec, x, y) AT2D(vec, dim, x, y)
    
    std::size_t dim = (unsigned int)(std::sqrt(img.size()) + 0.01);
    
    //Apply sobel operators
    std::vector<float> out_img_gx;
    GaussianBlurTool::conv_2d_square(sobel_gx, img, out_img_gx);
    
    std::vector<float> out_img_gy;
    GaussianBlurTool::conv_2d_square(sobel_gy, img, out_img_gy);
    
    //calculate direction of gradiant
    std::vector<int> out_img_gxy_dir;
    out_img_gxy_dir.resize(img.size(),-1);
    
    for (std::size_t i_x = 0; i_x <dim; ++i_x) {
        for (std::size_t i_y = 0; i_y <dim; ++i_y) {
            
            double ang = std::atan2(AT2D_(out_img_gy,i_x,i_y),AT2D_(out_img_gx,i_x,i_y))/std::acos(-1.);
            if (ang<0.) ang += 2.;
            
            int ang_i = 0*(ang <1/8. || ang >= 15/8.) + 180*(ang >= 7/8. && ang <9/8.) +
            45*(ang >= 1/8. && ang <3/8.) + 225*(ang >= 9/8. && ang <11/8. ) +
            90*(ang >= 3/8. && ang <5/8.) + 270*(ang >= 11/8. && ang <13/8. ) +
            135*(ang >= 5/8. && ang <7/8.) + 315*(ang >= 13/8. && ang <15/8. );
            
            if (std::fabs(AT2D_(out_img_gx,i_x,i_y)) <1.e-6 && std::fabs(AT2D_(out_img_gy,i_x,i_y)) <1.e-6 ) ang_i = -1;
            
            
            AT2D_(out_img_gxy_dir,i_x,i_y) = ang_i;
            
        } //i_y
    } //i_x
    
    
    //// non-max suppression..
    std::vector<int> out_img_max_binary;
    out_img_max_binary.resize(img.size(), -1);
    
    //// ..and tracing
    float low_threshold2 = GetLowHystThreshold2();
    float high_threshold2 = GetHighHystThreshold2();
    
    std::vector<std::pair<int,int> > out_img_tracing_seed_high;
    std::set<std::pair<int,int> > out_img_tracing_seed_low; //contains both > low and > high
    

#define GRAD2_(xy_, a_, b_) AT2D_(xy_, i_x+a_, i_y+b_) * AT2D_(xy_, i_x+a_, i_y+b_)
#define GRAD2(a_, b_) GRAD2_(out_img_gx, a_, b_) + GRAD2_(out_img_gy, a_, b_)

    //avoid most outwards pixels
    for (std::size_t i_x = 1; i_x < dim-1; ++i_x) {
        for (std::size_t i_y = 1; i_y < dim-1; ++i_y) {
            
            int ang_i =  AT2D_(out_img_gxy_dir,i_x,i_y);
            float grad2 = GRAD2(0,0);
            
            bool max = false;
            switch (ang_i) {
                case 0:
                case 180: //0 degree, horizontal
                {
                    float grad2_bef = GRAD2(-1, 0);
                    float grad2_aft = GRAD2(+1, 0);
                    if ( grad2 > grad2_bef && grad2 > grad2_aft) max = true;
                }
                    break;
                    
                case 45:
                case 225: //45 degree, diagonal
                {
                    float grad2_bef = GRAD2(-1, -1);
                    float grad2_aft = GRAD2(+1, +1);
                    if ( grad2 > grad2_bef && grad2 > grad2_aft) max = true;
                }
                    break;
                    
                case 90:
                case 270: //90 degree, vertical
                {
                    float grad2_bef = GRAD2(0, -1);
                    float grad2_aft = GRAD2(0, +1);
                    if ( grad2 > grad2_bef && grad2 > grad2_aft) max = true;
                }
                    break;
                    
                case 135:
                case 315: //135 degree, diagonal bis
                {
                    float grad2_bef = GRAD2(+1, -1);
                    float grad2_aft = GRAD2(-1, +1);
                    if ( grad2 > grad2_bef && grad2 > grad2_aft) max = true;
                }
                    break;
                    
                default:
                    //ang = -1, ignore
                    break;
            }
            
            AT2D_(out_img_max_binary,i_x,i_y) = max;
            
            //tracing
            if (!max) continue;
            
            if (grad2 > high_threshold2) out_img_tracing_seed_high.emplace_back(i_x,i_y);
            if (grad2 > low_threshold2) out_img_tracing_seed_low.emplace(i_x,i_y);

            
        } //i_y
    } //i_x


#undef GRAD2
#undef GRAD2_

    
    std::set<std::pair<int, int> > results;
    for (const auto & it_high :  out_img_tracing_seed_high ) {
        
        int i_x = it_high.first;
        int i_y = it_high.second;
        results.emplace(i_x, i_y);
        
        connect(i_x, i_y, out_img_tracing_seed_low, out_img_gxy_dir, results);
    }
    
    /////////////////////
    
    
    //copy result..
    
    std::vector<float> img_tmp;
    img_tmp.resize(img.size(),0.);
    for (const auto & it_res : results) {
       
        int x = it_res.first;
        int y = it_res.second;
        
        AT2D_(img_tmp,x,y) = 1.;
    
    }
    
    img.swap(img_tmp);
    
#undef AT2D_
}




//recursively connect points
void CannyEdgeTool::connect( int i_x, int i_y, const std::set<std::pair<int, int> >& low_hits, const std::vector<int>& direction, std::set< std::pair<int, int> > & results ) {
    
#define AT2D_(vec, x, y) AT2D(vec, dim, x, y)
#define INSERT_AND_CONNECT(next, a_, b_)  std::pair<int,int> next(i_x + a_, i_y + b_); \
                                          if ( low_hits.find(next) != low_hits.end() && results.find(next) == results.end() ) { \
                                             results.emplace(next.first, next.second); \
                                             connect(next.first, next.second, low_hits, direction, results); }
    
    std::size_t dim = (unsigned int)(std::sqrt(direction.size()) + 0.01);
    
    int ang_i = AT2D_(direction,i_x,i_y);
    
    switch (ang_i) {
        case 0:
        case 180: //0 degree, horizontal
        {
            //up
            INSERT_AND_CONNECT(up, 0, +1)
            
            //down
            INSERT_AND_CONNECT(down, 0, -1)
            
        }
            break;
            
        case 45:
        case 225: //45 degree, diagonal
        {
            //up
            INSERT_AND_CONNECT(up, +1, -1)
            
            //down
            INSERT_AND_CONNECT(down, -1, +1)
            
        }
            break;
            
        case 90:
        case 270: //90 degree, vertical
        {
            //up
            INSERT_AND_CONNECT(up, +1, 0)

            //down
            INSERT_AND_CONNECT(down, -1, 0)
            
        }
            break;
            
        case 135:
        case 315: //135 degree, diagonal bis
        {
            //up
            INSERT_AND_CONNECT(up, +1, +1)
            
            //down
            INSERT_AND_CONNECT(down, -1, -1)
                        
        }
            break;
            
        default:
            //ang = -1, ignore
            break;
    }

#undef INSERT_AND_CONNECT
#undef AT2D_
}


