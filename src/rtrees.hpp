///////////////
/* from 
https://raw.github.com/Itseez/opencv/2.4.6.1/modules/ml/src/rtrees.cpp
*/
///////////////


//#include "precomp.hpp"
//#include "opencv/ml.h"

#include <stdexcept>
#include <mutex>
#include <iomanip>

class MyCvForestTree;

class MyCvRTrees : public CvRTrees
{

  public:
    
     //default constructor
     MyCvRTrees() : CvRTrees(), i_thread(-1), console_offset_ptr(0) {};
    
     //thread aware
     MyCvRTrees(int i, const std::atomic<int>* off) : CvRTrees(), i_thread(i), console_offset_ptr(off) {};
  
  
     bool predict_prob_multiclass( const cv::Mat & , const cv::Mat &, std::vector< std::pair<float, float> >&) const;
    
     float get_class_value(int idx) const { return m_class_idx_map.at(idx); };
    
     friend class MyCvForestTree;
    
     friend class MLClassification;

  protected:
  
     static std::mutex m_log_mtx;
    
     void log_progress(int, int, int w = 50);
  
     int i_thread;
     const std::atomic<int>* console_offset_ptr;
  
     virtual bool grow_forest( const CvTermCriteria term_crit ); //overload
    
     std::vector<float> m_class_idx_map;

};

std::mutex MyCvRTrees::m_log_mtx;

void MyCvRTrees::log_progress(int x, int n, int w) {

  //sanity
  if (i_thread < 0 || !console_offset_ptr ) return;
  
  //increments of 1/100 only
  if ( ( x!=n) && (x % (n/100) != 0) ) return;

  float ratio = x/(float)n;
  int c = ratio * w;
  
  //lock mutex
  std::lock_guard<std::mutex> lock(m_log_mtx);
  
 
  std::cerr<<"\033["<<i_thread+ *console_offset_ptr<<"A\r";
  std::cerr<<std::setw(3)<<(int)(ratio*100)<<"% [";
  for (int j = 0; j < c-1; ++j) std::cerr<<"=";
  if (c < w && c>0) std::cerr<<">"; else if (c > 0) std::cerr<<"=";
  for (int j = c; j < w; ++j) std::cerr<<" ";
  std::cerr<<"]\033["<<i_thread+ *console_offset_ptr<<"B\r";
  

}

class MyCvForestTree: public CvForestTree
{

 public:
    inline const std::vector< int > & get_node_results(CvDTreeNode*) const;

 protected:
    virtual void calc_node_value( CvDTreeNode* node ); //overload
    
    //position in vector == class idx
    std::unordered_map< CvDTreeNode*, std::vector< int > > m_leaf_nodes_class_count;

};

const std::vector<  int > & MyCvForestTree::get_node_results(CvDTreeNode* node) const{

   return m_leaf_nodes_class_count.at(node);
     
}

void MyCvForestTree::calc_node_value( CvDTreeNode* node ) {
    
    //base
    CvForestTree::calc_node_value(node);
    
    //all nodes are considered.. we're growing the tree, don't know yet if a leaf node or not..
    if( data->is_classifier) {
        
        if ( m_leaf_nodes_class_count.find(node) != m_leaf_nodes_class_count.end()) {
            //std::cerr<<"leaf node "<< node <<"  found... wasn't expected."<<std::endl;
            //return;
            throw std::runtime_error("leaf node found... wasn't expected.");
        }
        
        auto& multi_class_count =  m_leaf_nodes_class_count[node];
        
        
        //cls_count buffer should already be filled by base class call to calc_node_value
        int* cls_count = data->counts->data.i;
        int m = data->get_num_classes();
        
       
        /*for( int k = 0; k < m; k++ )
            cls_count[k] = 0;
        
        int n = node->sample_count;
        int cv_n = data->params.cv_folds; //should be == 0 for RTrees (no tree pruning)
        
        int base_size = data->is_classifier ? m*cv_n*sizeof(int) : 2*cv_n*sizeof(double)+cv_n*sizeof(int);
        int ext_size = n*(sizeof(int) + (data->is_classifier ? sizeof(int) : sizeof(int)+sizeof(float)));
        cv::AutoBuffer<uchar> inn_buf(base_size + ext_size);
        uchar* base_buf = (uchar*)inn_buf;
        uchar* ext_buf = base_buf + base_size;
        
        int* cv_labels_buf = (int*)ext_buf;
        int* responses_buf = cv_labels_buf + n;
        const int* responses = data->get_class_labels(node, responses_buf);
        
        
        
        if( cv_n == 0 )
        {
            for( int i = 0; i < n; i++ )
                cls_count[responses[i]]++;
        }
        */
        
        bool first = false;
        MyCvRTrees *the_forest = dynamic_cast<MyCvRTrees*>(forest);
        if (!the_forest) {
           //std::cerr<<"can't dynamic cast forest"<<std::endl; return;
           throw std::runtime_error("can't dynamic cast forest");
           }
        
        if (the_forest->m_class_idx_map.size() == 0) {
             first = true;
             the_forest->m_class_idx_map.resize(m);
        }
        
        for( int k = 0; k < m; k++ ) {
            
            int count = cls_count[k];
            multi_class_count.push_back(count);
            
            if (first)
              the_forest->m_class_idx_map.at(k) = data->cat_map->data.i[data->cat_ofs->data.i[data->cat_var_count] + k];
            
            
        }
        
        
    }
}

////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////


bool MyCvRTrees::predict_prob_multiclass( const cv::Mat & sample , const cv::Mat & missing,
                                           std::vector< std::pair<float, float> >& results_probabilities) const {
    //classification only
    if (nclasses<=0) return false;
    
    results_probabilities.clear();
    results_probabilities.resize(nclasses);
    for (auto & res : results_probabilities) {
        res.first = -1.;
        res.second = 0.;
    }
        
    
    
    for( int k = 0; k < ntrees; ++k ) {
    
        CvDTreeNode* predicted_node = trees[k]->predict( sample, missing );
        const MyCvForestTree * f_tree = dynamic_cast<const MyCvForestTree *>(trees[k]);
        if (!f_tree) {
          //std::cerr<<" couldn't dynamic_cast MyCvForestTree"<<std::endl;
          //return false;
           throw std::runtime_error("couldn't dynamic_cast MyCvForestTree");
        }
        
        
        //sanity
        try {
            f_tree->get_node_results(predicted_node);
        } catch (const std::out_of_range& oor) {
            //std::cerr << "Out of Range error: " << oor.what() << '\n';
            //return false;
            throw std::runtime_error("Out of Range error in get_node_results");
        }
    
        const std::vector< int > & res = f_tree->get_node_results(predicted_node);
        //sanity
        if ((int)res.size() != nclasses) {
           //std::cerr<<"size mismatch classes"<<std::endl;
           //return false;
           throw std::runtime_error("size mismatch classes");
        }
        
        int total = 0;
        for (auto const & counts : res )
             total += counts;
        
        
        
        for (auto it_counts = res.cbegin(); it_counts != res.cend(); ++it_counts ) {
             std::size_t dist = std::distance(res.cbegin(), it_counts);
             int num =  *it_counts;
            
             //sanity
             if (total == 0) {
                //std::cerr<<" What ? 0 counts in node.."<<std::endl;
                throw std::runtime_error("What ? 0 counts in node..");
             }
             //relative frequency (//Laplace estimate)
             results_probabilities.at(dist).second += (float)(num)/(total); // (float)(1. + num)/(nclasses + total);
            
            if (k == ntrees-1) {
              //fill in value and normalize
              results_probabilities.at(dist).second /= ntrees;
              
              float val = -1.;
              try { val = get_class_value(dist); }
              catch (const std::out_of_range& oor) {
                //std::cerr << "Out of Range error: " << oor.what() << std::endl;
                throw std::runtime_error("Out of Range error: in get_class_value");
               }

              results_probabilities.at(dist).first = val;
            
            }
        }
        
        
        
    }
    
    
    return true;

}

//not much choice but to re_implement grow_forest.
// reason is training data kept in buffers (cvDtreeTrainingdata) that are only valid for the last processed tree.
// so need to get the results (counts per class) in the (overloaded) calc_node_value method

bool MyCvRTrees::grow_forest( const CvTermCriteria term_crit )
{

    CvMat* sample_idx_mask_for_tree = 0;
    CvMat* sample_idx_for_tree      = 0;

    const int max_ntrees = term_crit.max_iter;
    const double max_oob_err = term_crit.epsilon;

    const int dims = data->var_count;
    float maximal_response = 0;

    CvMat* oob_sample_votes    = 0;
    CvMat* oob_responses       = 0;

    float* oob_samples_perm_ptr= 0;

    float* samples_ptr     = 0;
    uchar* missing_ptr     = 0;
    float* true_resp_ptr   = 0;
    bool is_oob_or_vimportance = (max_oob_err > 0 && term_crit.type != CV_TERMCRIT_ITER) || var_importance;

    // oob_predictions_sum[i] = sum of predicted values for the i-th sample
    // oob_num_of_predictions[i] = number of summands
    //                            (number of predictions for the i-th sample)
    // initialize these variable to avoid warning C4701
    CvMat oob_predictions_sum = cvMat( 1, 1, CV_32FC1 );
    CvMat oob_num_of_predictions = cvMat( 1, 1, CV_32FC1 );

    nsamples = data->sample_count;
    nclasses = data->get_num_classes();

    if ( is_oob_or_vimportance )
    {
        if( data->is_classifier )
        {
            oob_sample_votes = cvCreateMat( nsamples, nclasses, CV_32SC1 );
            cvZero(oob_sample_votes);
        }
        else
        {
            // oob_responses[0,i] = oob_predictions_sum[i]
            //    = sum of predicted values for the i-th sample
            // oob_responses[1,i] = oob_num_of_predictions[i]
            //    = number of summands (number of predictions for the i-th sample)
            oob_responses = cvCreateMat( 2, nsamples, CV_32FC1 );
            cvZero(oob_responses);
            cvGetRow( oob_responses, &oob_predictions_sum, 0 );
            cvGetRow( oob_responses, &oob_num_of_predictions, 1 );
        }

        oob_samples_perm_ptr     = (float*)cvAlloc( sizeof(float)*nsamples*dims );
        samples_ptr              = (float*)cvAlloc( sizeof(float)*nsamples*dims );
        missing_ptr              = (uchar*)cvAlloc( sizeof(uchar)*nsamples*dims );
        true_resp_ptr            = (float*)cvAlloc( sizeof(float)*nsamples );

        data->get_vectors( 0, samples_ptr, missing_ptr, true_resp_ptr );

        double minval, maxval;
        CvMat responses = cvMat(1, nsamples, CV_32FC1, true_resp_ptr);
        cvMinMaxLoc( &responses, &minval, &maxval );
        maximal_response = (float)MAX( MAX( fabs(minval), fabs(maxval) ), 0 );
    }

    trees = (CvForestTree**)cvAlloc( sizeof(trees[0])*max_ntrees );
    memset( trees, 0, sizeof(trees[0])*max_ntrees );

    sample_idx_mask_for_tree = cvCreateMat( 1, nsamples, CV_8UC1 );
    sample_idx_for_tree      = cvCreateMat( 1, nsamples, CV_32SC1 );

    ntrees = 0;
    while( ntrees < max_ntrees )
    {
    
        log_progress(ntrees, max_ntrees);
        
        
        int i, oob_samples_count = 0;
        double ncorrect_responses = 0; // used for estimation of variable importance
        CvForestTree* tree = 0;

        cvZero( sample_idx_mask_for_tree );
        for(i = 0; i < nsamples; i++ ) //form sample for creation one tree
        {
            int idx = (*rng)(nsamples);
            sample_idx_for_tree->data.i[i] = idx;
            sample_idx_mask_for_tree->data.ptr[idx] = 0xFF;
        }

        trees[ntrees] = new MyCvForestTree();
        tree = trees[ntrees];
        tree->train( data, sample_idx_for_tree, this );

        if ( is_oob_or_vimportance )
        {
            CvMat sample, missing;
            // form array of OOB samples indices and get these samples
            sample   = cvMat( 1, dims, CV_32FC1, samples_ptr );
            missing  = cvMat( 1, dims, CV_8UC1,  missing_ptr );

            oob_error = 0;
            for( i = 0; i < nsamples; i++,
                sample.data.fl += dims, missing.data.ptr += dims )
            {
                CvDTreeNode* predicted_node = 0;
                // check if the sample is OOB
                if( sample_idx_mask_for_tree->data.ptr[i] )
                    continue;

                // predict oob samples
                if( !predicted_node )
                    predicted_node = tree->predict(&sample, &missing, true);

                if( !data->is_classifier ) //regression
                {
                    double avg_resp, resp = predicted_node->value;
                    oob_predictions_sum.data.fl[i] += (float)resp;
                    oob_num_of_predictions.data.fl[i] += 1;

                    // compute oob error
                    avg_resp = oob_predictions_sum.data.fl[i]/oob_num_of_predictions.data.fl[i];
                    avg_resp -= true_resp_ptr[i];
                    oob_error += avg_resp*avg_resp;
                    resp = (resp - true_resp_ptr[i])/maximal_response;
                    ncorrect_responses += exp( -resp*resp );
                }
                else //classification
                {
                    double prdct_resp;
                    CvPoint max_loc;
                    CvMat votes;

                    cvGetRow(oob_sample_votes, &votes, i);
                    votes.data.i[predicted_node->class_idx]++;

                    // compute oob error
                    cvMinMaxLoc( &votes, 0, 0, 0, &max_loc );

                    prdct_resp = data->cat_map->data.i[max_loc.x];
                    oob_error += (fabs(prdct_resp - true_resp_ptr[i]) < FLT_EPSILON) ? 0 : 1;

                    ncorrect_responses += cvRound(predicted_node->value - true_resp_ptr[i]) == 0;
                }
                oob_samples_count++;
            }
            if( oob_samples_count > 0 )
                oob_error /= (double)oob_samples_count;

            // estimate variable importance
            if( var_importance && oob_samples_count > 0 )
            {
                int m;

                memcpy( oob_samples_perm_ptr, samples_ptr, dims*nsamples*sizeof(float));
                for( m = 0; m < dims; m++ )
                {
                    double ncorrect_responses_permuted = 0;
                    // randomly permute values of the m-th variable in the oob samples
                    float* mth_var_ptr = oob_samples_perm_ptr + m;

                    for( i = 0; i < nsamples; i++ )
                    {
                        int i1, i2;
                        float temp;

                        if( sample_idx_mask_for_tree->data.ptr[i] ) //the sample is not OOB
                            continue;
                        i1 = (*rng)(nsamples);
                        i2 = (*rng)(nsamples);
                        CV_SWAP( mth_var_ptr[i1*dims], mth_var_ptr[i2*dims], temp );

                        // turn values of (m-1)-th variable, that were permuted
                        // at the previous iteration, untouched
                        if( m > 1 )
                            oob_samples_perm_ptr[i*dims+m-1] = samples_ptr[i*dims+m-1];
                    }

                    // predict "permuted" cases and calculate the number of votes for the
                    // correct class in the variable-m-permuted oob data
                    sample  = cvMat( 1, dims, CV_32FC1, oob_samples_perm_ptr );
                    missing = cvMat( 1, dims, CV_8UC1, missing_ptr );
                    for( i = 0; i < nsamples; i++,
                        sample.data.fl += dims, missing.data.ptr += dims )
                    {
                        double predct_resp, true_resp;

                        if( sample_idx_mask_for_tree->data.ptr[i] ) //the sample is not OOB
                            continue;

                        predct_resp = tree->predict(&sample, &missing, true)->value;
                        true_resp   = true_resp_ptr[i];
                        if( data->is_classifier )
                            ncorrect_responses_permuted += cvRound(true_resp - predct_resp) == 0;
                        else
                        {
                            true_resp = (true_resp - predct_resp)/maximal_response;
                            ncorrect_responses_permuted += exp( -true_resp*true_resp );
                        }
                    }
                    var_importance->data.fl[m] += (float)(ncorrect_responses
                        - ncorrect_responses_permuted);
                }
            }
        }
        ntrees++;
        if( term_crit.type != CV_TERMCRIT_ITER && oob_error < max_oob_err )
            break;
    }
    
    //done
    log_progress(max_ntrees, max_ntrees);

    if( var_importance )
    {
        for ( int vi = 0; vi < var_importance->cols; vi++ )
                var_importance->data.fl[vi] = ( var_importance->data.fl[vi] > 0 ) ?
                    var_importance->data.fl[vi] : 0;
        cvNormalize( var_importance, var_importance, 1., 0, CV_L1 );
    }

    cvFree( &oob_samples_perm_ptr );
    cvFree( &samples_ptr );
    cvFree( &missing_ptr );
    cvFree( &true_resp_ptr );

    cvReleaseMat( &sample_idx_mask_for_tree );
    cvReleaseMat( &sample_idx_for_tree );

    cvReleaseMat( &oob_sample_votes );
    cvReleaseMat( &oob_responses );

    return true;
}


