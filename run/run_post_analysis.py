import sys
from Services.python.IncidentService import IncidentService
from UtilityTools.python.RootNtupleTools import RootNtupleReaderTool, intp_value

from exceptions import BaseException


def main():


  # Reader tool : fetches the prediction probabilities vectors from the ROOT file
  root_svc_reader = RootNtupleReaderTool("RootToolReader","tree_results.root", "test/ttree", 2)
  
  
  accuracy = 0.
  accuracy_twohighest = 0.;
  accuracy_threshold = [0.]*100
  ientry_threshold = [0.]*100
    
  # loop over entries/events in input TTree
  ientry = 0;
  while True:
    try:
      vec = root_svc_reader.GetBranchEntry_FloatVector("prediction_prob",ientry)
      targ = root_svc_reader.GetBranchEntry_Int("target", ientry)
    except BaseException as e:
      print "Cauth Error! -> ", str(e)
      break

    # if EOF reached, stop.
    if targ == None: break;
    
    #make a tuple out of it:
    vec_tuple = []
    for i in range(0,10):
        vec_tuple += [(i, vec[i])]
    
    #sort
    vec_tuple.sort(key=lambda pr: pr[1], reverse=True)

    # check if largest prob. prediction is accurate
    if intp_value(targ) ==  vec_tuple[0][0] : accuracy += 1.
    
    # check is one of the two largest predictions is accurate
    if intp_value(targ) ==  vec_tuple[0][0] or intp_value(targ) ==  vec_tuple[1][0]:
       accuracy_twohighest += 1.;
       
    # check if largest prob. prediction is accurate
    #  but only consider those with > x% = th probability
    for i in range(0,100):
       th = i/100.
       if vec_tuple[0][1] > th:
          ientry_threshold[i] += 1.
          if intp_value(targ) ==  vec_tuple[0][0]:
             accuracy_threshold[i] += 1.
    
    ientry += 1
    #if (ientry == 1): break

  print "read ", ientry, " entries"
  
  print "Success rate: ", 100.*(accuracy/ientry), " %"
  print "Success rate (two highest): ", 100.*(accuracy_twohighest/ientry), " %"
  #print "Success rate (thresholds): "
  #for  i in range(0,10):
  #     print "  ",i," ", 100.*(accuracy_threshold[i]/ientry_threshold[i]), " %"," with ", 100.*(ientry_threshold[i]/ientry), " % acceptance"
  

  # ROOT imports
  from ROOT import ROOT, TFile, TGraph
  f = TFile.Open("post_results.root", "RECREATE")
  gr = TGraph(100)
  for  i in range(0,100):
      gr.SetPoint(i, 100.*(accuracy_threshold[i]/ientry_threshold[i]),100.*(ientry_threshold[i]/ientry) )

  gr.Write("accuracy_vs_acceptance")

  return 0

if __name__ == '__main__':
    # main should return 0 for success, something else (usually 1) for error.
    sys.exit(main())


