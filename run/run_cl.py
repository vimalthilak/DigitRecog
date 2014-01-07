import sys
from Services.python.IncidentService import IncidentService, Incident
from DigitRecog.python.DigitRecog import MLClassification
from UtilityTools.python.RootNtupleTools import RootNtupleWriterTool, RootNtupleReaderTool, intp_value

from exceptions import BaseException



def main():

  num = 42000

  # Reader tool : fetches the features vectors from the ROOT file
  # (one vector of features per event)
  root_svc_reader = RootNtupleReaderTool("RootToolReader","tree.root", "ttree", 2)
  
  # Random Forest Classifier tool
  ml = MLClassification("mlTool",2)
  
  # loop over entries/events in input TTree and feed them to the mlTool
  ientry = 0;
  while True:
    try:
      vec = root_svc_reader.GetBranchEntry_DoubleVector("features",ientry)
      targ = root_svc_reader.GetBranchEntry_Int("target", ientry)
    except BaseException as e:
      print "Cauth Error! -> ", str(e)
      break

    # if EOF reached, stop.
    if targ == None: break;

    # Use 'num/2' events for the training process
    # remaining 'num/2' is for testing
    try:
      if ientry < num/2 : ml.accumulateTrain(intp_value(targ), vec)
      else : ml.accumulateTest(intp_value(targ), vec)
    except BaseException as e:
      print "Cauth Error! -> ", str(e)
      break

    ientry += 1
    if (ientry == num): break

  print "read ", ientry, " entries"
  
  # sanity check;
  #   make sure the desired amount of events has been processed before continuing
  if ( ientry != num): return 1
  
  
  inc_svc = IncidentService.getInstance()
  
  # results from the training (cross-validation) process are output to a TTree
  root_svc = RootNtupleWriterTool("RootTool", "tree_results.root", "train/ttree", 2)
  ml.setRootNtupleHelper(root_svc)
  
  inc_svc.fireIncident(Incident("BeginRun"))
  
  #####
  # Performs the cross-validation (using 500 trees by default)
  #####
  try:
    ml.performCrossValidationTraining(4,    # number of c.-v. regions (multithreaded)
                                      25,   # max depth of trees
                                       5,   # min sample count in leaves
                                      20)   # number of variables (features) used per node
  except BaseException as e:
    print "Cauth Error! -> ", str(e)
    inc_svc.fireIncident(Incident("EndRun"))
    return 1

  root_svc.stop() #stop listening to Begin/EndEvent

  
  #####
  # Performs the training process on the whole training dataset
  #####
  try:
    ml.performTraining(25, 5, 20)
  except BaseException as e:
    print "Cauth Error! -> ", str(e)
    inc_svc.fireIncident(Incident("EndRun"))
    return 1

  
  # results from the testing procedure are saved in a TTree
  root_svc_test = RootNtupleWriterTool("RootToolTest", "tree_results.root", "test/ttree", 2)
  ml.setRootNtupleHelper(root_svc_test)
  
  inc_svc.fireIncident(Incident("BeginRun"))

  #####
  # Performs the testing process
  #####
  try:
    ml.performTesting("train")
  except BaseException as e:
    print "Cauth Error! -> ", str(e)
    inc_svc.fireIncident(Incident("EndRun"))
    return 1

  #################

  # we're done, save output
  inc_svc.fireIncident(Incident("EndRun"))
  inc_svc.kill()
  


  return 0

if __name__ == '__main__':
    # main should return 0 for success, something else (usually 1) for error.
    sys.exit(main())


