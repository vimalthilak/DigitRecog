import sys
from Services.python.IncidentService import IncidentService
from UtilityTools.python.RootNtupleTools import RootNtupleReaderTool, intp_value
from Services.python.Messaging import PyMessaging, logDEBUG, logINFO, logWARNING, logERROR

from exceptions import BaseException
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats.distributions as dist

def BayesDivide(pass_h, total_h):
   k = pass_h[0]
   n = total_h[0]

   # Bayesian statistics (http://root.cern.ch/root/html/TEfficiency.html) 
   # Beta(alpha=1, beta=1) prior (uniform)
   # efficiency estimator = expectation value of posterior prob: (k+alpha)/(N+alpha+beta)

   
   eff = (k+1)/(n+2) #np.where(n > 0, k/n, 0.5)

   # error calculation, 68.3% central interval (http://arxiv.org/pdf/1012.0566v3.pdf)
   c = 0.683
   p_lower = eff - dist.beta.ppf((1-c)/2.,k+1,n-k+1)
   p_upper = dist.beta.ppf(1-(1-c)/2.,k+1,n-k+1) - eff
   
   return eff, p_lower, p_upper

def main():

  #Messaging helper tool
  log = PyMessaging("main", logDEBUG)
  toLog = log.PyLOG
  
  # Reader tool : fetches the prediction probabilities vectors from the ROOT file
  # looking at cross-validation results
  root_svc_reader = RootNtupleReaderTool("RootToolReader","tree_results_.root", "train/ttree", 2)
  
  
  accuracy = 0.
  accuracy_per_cv = [0.]*4
  ientry_per_cv = [0]*4
  
  pred_proba = np.zeros(shape=(21000,10), dtype=float)
  targets = np.zeros(pred_proba.shape[0], dtype=int)
  
  # loop over entries/events in input TTree
  ientry = 0;
  while True:
    try:
      vec = root_svc_reader.GetBranchEntry_FloatVector("prediction_prob",ientry)
      targ = root_svc_reader.GetBranchEntry_Int("target", ientry)
      cv = root_svc_reader.GetBranchEntry_Int("CVregion", ientry)
    except BaseException as e:
      toLog("Cauth Error! -> "+ str(e), logERROR)
      break

    # if EOF reached, stop.
    if targ == None: break;
    
    pred_proba[ientry] = vec
    targets[ientry] = intp_value(targ)
    
    #make a tuple out of it:
    vec_tuple = []
    for i in range(0,10):
        vec_tuple += [(i, vec[i])]
    
    #sort
    vec_tuple.sort(key=lambda pr: pr[1], reverse=True)

    #sanity
    assert vec_tuple[0][0] == pred_proba[ientry].argmax()

    # check if largest prob. prediction is accurate
    if intp_value(targ) ==  vec_tuple[0][0] :
       accuracy += 1.
       # ditto. Per CV region
       accuracy_per_cv[intp_value(cv)] += 1.
    
    ientry_per_cv[intp_value(cv)] += 1
    
    ientry += 1
    #if (ientry == 1): break

  toLog( "read "+ str(ientry)+ " entries", logINFO)
  
  toLog( "Success rate (total): " + str(100.*(accuracy/ientry)) +  " %", logINFO)
  toLog( "Success rates (per CV region): \n" +
         "".join(["\t"+ str(x) + "-> "+ str(100.*accuracy_per_cv[x]/ientry_per_cv[x]) + " %\n" for x in xrange(0,4)])[:-1], logINFO)

  #########
  
  f, axarr = plt.subplots(2,5, sharey='row')

  hists = [axarr.flat[i].hist(pred_proba[:,i], bins=21, range=(-0.025, 1.025), histtype='step') for i in xrange(0,10)]
  for i in xrange(0,10): axarr.flat[i].set_title("Digit "+str(i))
  
  # only selects the ones where the target is the given digit under consideration
  # (boolean indexing)
  f_g, axarr_g = plt.subplots(2,5, sharey='row')
  hists_g = [axarr_g.flat[i].hist(pred_proba[:,i][targets==i], bins=21, range=(-0.025, 1.025), histtype='step') for i in xrange(0,10)]
  for i in xrange(0,10): axarr_g.flat[i].set_title("Digit "+str(i) + " - G")
  
  
  eff_bayes = [BayesDivide(hists_g[i], hists[i]) for i in xrange(0,10)]
  
  f_eff, axarr_eff = plt.subplots(2,5, sharey='row')
  hists_eff = [axarr_eff.flat[i].errorbar(np.arange(-0.025+0.05/2., 1.025, 0.05), eff_bayes[i][0], fmt="o",
               xerr=0.05/2., yerr=[eff_bayes[i][1],eff_bayes[i][2]],  markersize=4) for i in xrange(0,10)]
        
  for i in xrange(0,10): axarr_eff.flat[i].set_title("Digit "+str(i))
        
  
  plt.show()
  

  return 0

if __name__ == '__main__':
    # main should return 0 for success, something else (usually 1) for error.
    sys.exit(main())


