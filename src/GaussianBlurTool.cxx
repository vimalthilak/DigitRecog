#include "DigitRecog/GaussianBlurTool.h"
#include <cmath>
 
void GaussianBlurTool::execute(std::vector<float> & img) {

   
   std::vector<float> res;
   conv_2d_square(gaussian_filter_2, img, res);
   
   //copy result.. 
   img.swap(res);

}


//primitive 2d convolution implementation
//source: http://www.songho.ca/dsp/convolution/convolution.html#convolution_2d
template<std::size_t N_K>
void GaussianBlurTool::conv_2d_square(const float kernel[][N_K], const std::vector<float>& in_img,
                                                             std::vector<float>& out_img) {
    //allocate mem in advance
    out_img.resize(in_img.size(), 0);
    std::size_t dim = (unsigned int)(std::sqrt(in_img.size()) + 0.01);
    
    // find center position of kernel (half of kernel size)
    int kCenterX = N_K / 2;
    int kCenterY = N_K / 2;
    int rows = dim;
    int cols = dim;
    int kRows = N_K;
    int kCols = N_K;
    
    #define AT2D_(vec, x, y) AT2D(vec, cols, x, y)
    
    for(int i=0; i < rows; ++i)              // rows
    {
        for(int j=0; j < cols; ++j)          // columns
        {
            float sum = 0;                     // init to 0 before sum
            
            for(int m=0; m < kRows; ++m)     // kernel rows
            {
                int mm = kRows - 1 - m;      // row index of flipped kernel
                
                for(int n=0; n < kCols; ++n) // kernel columns
                {
                    int nn = kCols - 1 - n;  // column index of flipped kernel
                    
                    // index of input signal, used for checking boundary
                    int ii = i + (m - kCenterY);
                    int jj = j + (n - kCenterX);
                    
                    // ignore input samples which are out of bound
                    if( ii >= 0 && ii < rows && jj >= 0 && jj < cols )
                        sum += AT2D_(in_img, ii, jj) * kernel[mm][nn];
                }
            }
            
            AT2D_(out_img,i,j) = sum;
        }
    }

    #undef AT2D_
    
}

