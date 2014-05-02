import numpy as np
from Services.python.Messaging import PyMessaging, logINFO, logWARNING, logDEBUG, logERROR
from Services.python.IncidentService import PyIIncidentListener, Incident, IncidentService
from UtilityTools.python.RootNtupleTools import IRootNtupleWriterTool

from exceptions import BaseException, Exception

def accepts(offset, *types):
    def check_accepts(f):
        #executed when decorating a function
        assert len(types) == f.func_code.co_argcount - offset
        def new_f(*args, **kwds):
            #wrapper for decorated function
            for (a, t) in zip(args[offset:], types):
                assert isinstance(a, t), "arg %r does not match %s" % (a,t)
            return f(*args, **kwds)
        new_f.func_name = f.func_name
        return new_f
    return check_accepts

def count_calls(fn):
    def wrapper(*args, **kwargs):
        try:
          getattr(args[0],wrapper.__name__ + "_called")
        except AttributeError:
          setattr(args[0],wrapper.__name__ + "_called", 0)
       
        c = getattr(args[0],wrapper.__name__ + "_called")
        setattr(args[0],wrapper.__name__ + "_called",c+1)
        return fn(*args, **kwargs)
    wrapper.__name__= fn.__name__
    return wrapper

class ClassificationTool(PyMessaging, PyIIncidentListener):

  def __init__(self, n, arg1, arg2, lvl = logINFO):
    #super(ClassificationTool, self).__init__(n)
    PyMessaging.__init__(self, n, lvl)
    PyIIncidentListener.__init__(self)
    self._initData(arg1, arg2)
    
    #register handle
    inc_svc = IncidentService.getInstance()
    inc_svc.addListener(self, "BeginRun");

  @accepts(1,int,int)
  def _initData(self,s1, s2):
    self._train_size = s1
    self._train_target = np.zeros(shape=s1, dtype=int)
    
    self._test_size = s2
    self._test_target = np.zeros(shape=s2, dtype=int)
    
  @count_calls
  def accumulateData(self, targ, vec):
    index = self.accumulateData_called - 1
    
    #create arrays for train/test data
    if index == 0:
      self._train_features = np.zeros(shape=(self._train_size, vec.size()), dtype=np.float32)
      self._train_features_max = np.zeros(shape=vec.size(),dtype=np.float32)
      self._train_features_max.fill(-1.e99)
      self._train_features_min = np.zeros(shape=vec.size(),dtype=np.float32)
      self._train_features_min.fill(1.e99)
      self.PyLOG("Allocated matrix for Train dataset", logINFO)
    elif index == self._train_size:
      self._test_features = np.zeros(shape=(self._test_size, vec.size()), dtype=np.float32)
      self.PyLOG("Allocated matrix for Test dataset", logINFO)
    
    #train data set
    if index < self._train_size:
      assert vec.size() == self._train_features.shape[1]
      self._train_features[index] = vec
      self._train_target[index] = targ
      #update min/max vectors
      self._train_features_max = np.where(self._train_features[index] > self._train_features_max, self._train_features[index], self._train_features_max )
      self._train_features_min = np.where(self._train_features[index] < self._train_features_min, self._train_features[index], self._train_features_min )
      
    #test data set
    elif index < self._train_size + self._test_size:
      assert vec.size() == self._test_features.shape[1]
      self._test_features[index-self._train_size] = vec
      self._test_target[index-self._train_size] = targ
    else:
      raise Exception("Trying to accumulate too much data!")

  @accepts(1, IRootNtupleWriterTool)
  def setRootNtupleHelper(self, helper):
     self.ntuple_helper = helper

  def Initialize(self):
    raise Exception("test")
    pass
    

  def handle(self,incident):
     self.PyLOG("Handling incident ==" + incident.svcType() + "==", logDEBUG)
     if  incident.svcType() == "BeginRun" :
       try:
         self.Initialize()
       except BaseException as e:
         self.PyLOG("Couldn't initialize properly.. " + str(e), logERROR, True)

 
  #@property
  def train_features(self):
    return self._train_features

  def _scale(self, dataset):
    if hasattr(self, dataset.func_name +  "_scaled"):
      self.PyLOG("Trying to scale same dataset twice, ignoring", logWARNING)
      return

    try:
      data = dataset()
      data -= self._train_features_min
      data /= (self._train_features_max - self._train_features_min)
      data *= 2.
      data -= 1.
    except BaseException as e:
      self.PyLOG("Problem with scaling.. " + str(e), logERROR)
      return

    setattr(self, dataset.func_name +  "_scaled", True)






