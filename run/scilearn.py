import numpy as np
from simplefwk_services.python.Messaging import PyMessaging, logINFO, logWARNING, logDEBUG, logERROR
from simplefwk_services.python.IncidentService import PyIIncidentListener, Incident, IncidentService
from simplefwk_utilitytools.python.RootNtupleTools import IRootNtupleWriterTool, any, int_p, float_p
from simplefwk_utilitytoolsinterfaces.python.ObjectHolder import VectorObjectHolder_Float, SingleObjectHolder_Int



from exceptions import BaseException, Exception

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn import cross_validation
from sklearn.base import clone
from sklearn.metrics import accuracy_score

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
    v = VectorObjectHolder_Float()
    v.thisown = 0 #do not acquire ownership
    self.ntuple_helper.registerBranch("prediction_prob", v)
    
    v = SingleObjectHolder_Int(-1)
    v.thisown = 0
    self.ntuple_helper.registerBranch("target", v)
    
    v = SingleObjectHolder_Int(-1)
    v.thisown = 0
    self.ntuple_helper.registerBranch("prediction", v)
    
    v = SingleObjectHolder_Int(-1)
    v.thisown = 0
    self.ntuple_helper.registerBranch("CVregion", v)
    

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
      #not optimal...
      data = dataset()
      data -= self._train_features_min
      data /= (self._train_features_max - self._train_features_min)
      data *= 2.
      data -= 1.
    except BaseException as e:
      self.PyLOG("Problem with scaling.. " + str(e), logERROR)
      return

    setattr(self, dataset.func_name +  "_scaled", True)

  @accepts(1,int,int,int,int,int)
  def performCrossValidationTraining(self, num_cv=4, max_depth=25, min_count_leaf=5, num_features=20, num_trees=500):
    #scale first
    self._scale(self.train_features)

    clf = RFC(n_jobs=4, random_state=112, n_estimators=num_trees, min_samples_leaf=min_count_leaf, max_features=num_features, max_depth=max_depth)
    #scores = cross_validation.cross_val_score(clf, self._train_features, self._train_target, cv=num_cv, n_jobs=4, verbose=2)
    #self.PyLOG(str(scores), logDEBUG)
    kf = cross_validation.KFold(self._train_target.shape[0], n_folds=4,shuffle=True, random_state=1234)

    cv_n = int_p()
    cv_n.assign(-1)
    for train, test in kf:
       cv_n.assign(cv_n.value() + 1)
       
       x_train = self._train_features[train]
       y_train = self._train_target[train]

       x_test = self._train_features[test]
       y_test = self._train_target[test]

       clf_clone = clone(clf)
       clf_clone.fit(x_train, y_train)

       y_test_predict_proba = clf_clone.predict_proba(x_test)
       y_test_predict = clf_clone.classes_.take(np.argmax(y_test_predict_proba, axis=1), axis=0)
       self.PyLOG("acc: "+str(accuracy_score(y_test, y_test_predict)), logINFO )

       assert clf_clone.n_classes_ == np.unique(clf_clone.classes_).shape[0]
       y_test_matrix = np.zeros(shape=(y_test.shape[0], clf_clone.n_classes_))
       for i in xrange(0,y_test.shape[0]):
         y_test_matrix[i][np.where(clf_clone.classes_ == y_test[i])[0][0]] = 1.

       #brier score
       assert y_test_matrix.shape == y_test_predict_proba.shape
       b = np.power( y_test_predict_proba - y_test_matrix , 2)
       self.PyLOG("brier score: "+str(np.sum(b)/y_test.shape[0]), logINFO)

       inc_svc = IncidentService.getInstance()

       tar = int_p()
       pred = int_p()
       pred_prob = float_p()
       
       for i in xrange(0,y_test.shape[0]):
         inc_svc.fireIncident(Incident("BeginEvent"))

         self.ntuple_helper.pushBack("CVregion", any(cv_n))
         
         tar.assign(y_test[i])
         self.ntuple_helper.pushBack("target", any(tar))
         
         pred.assign(y_test_predict[i])
         self.ntuple_helper.pushBack("prediction", any(pred))
         
         for j in  xrange(0,y_test_predict_proba.shape[1]):
           assert j == np.where(clf_clone.classes_ == j)[0][0]
           pred_prob.assign(y_test_predict_proba[i][j])
           self.ntuple_helper.pushBack("prediction_prob", any(pred_prob))
         

         inc_svc.fireIncident(Incident("EndEvent"))

       
       





