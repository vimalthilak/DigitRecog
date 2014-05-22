import sys
from simplefwk_services.python.IncidentService import IncidentService, Incident
from DigitRecog.python.DigitRecog import FeX , GaussianBlurTool, CannyEdgeTool , FloatVector
from simplefwk_utilitytools.python.RootNtupleTools import RootNtupleWriterTool

from exceptions import BaseException

import csv

def _run(tool):

  inc_svc = IncidentService.getInstance()
  inc_svc.fireIncident(Incident("BeginRun"))

  # change input file location
  with open("train.csv", "r") as f:
     csv_reader = csv.reader(f)
     for row in csv_reader:
        row_int = map(float,row) #convert to list of floats (from list of string)
  
        inc_svc.fireIncident(Incident("BeginEvent"))
      
        digit_class = int(row[0])
        img_vec = FloatVector(row_int[1:])

        try:
           tool.execute(digit_class, img_vec)
        except BaseException as e:
           print "Caugth Error! -> ", str(e)


        inc_svc.fireIncident(Incident("EndEvent"))
        
        if csv_reader.line_num % 5000  == 0: print "..processed ", csv_reader.line_num, " lines"
 

  inc_svc.fireIncident(Incident("EndRun"))

def main():

  # Ntuple Writer tools:
  #  - One for the features per se
  #  - One containing the names (strings) of the features
  root_svc = RootNtupleWriterTool("RootTool", "tree.root", "ttree")
  root_svc_meta = RootNtupleWriterTool("RootToolMeta", "tree.root", "ttree_meta", 2, True) # only one event
  
  # Pre-processing tools
  gauss = GaussianBlurTool()
  canny = CannyEdgeTool()
  
  # Feature Extraction Algorithm
  fex = FeX("myFex", 2, root_svc, root_svc_meta)
  
  fex.addPreProcessorTool(gauss)
  fex.addPreProcessorTool(canny)
   
  _run(fex)
  
  inc_svc = IncidentService.getInstance()
  inc_svc.kill()
  

  return 0

if __name__ == '__main__':
    # main should return 0 for success, something else (usually 1) for error.
    sys.exit(main())


