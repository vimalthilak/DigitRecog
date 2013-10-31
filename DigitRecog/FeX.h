#include "Services/Messaging.h"
#include "Services/IIncidentListener.h"
#include <string>
#include <vector>

class IRootNtupleWriterTool;
class IPreProcessorTools;

class FeX: public Messaging, virtual public IIncidentListener
{
public:
 

  FeX(const std::string& n, IRootNtupleWriterTool *, IRootNtupleWriterTool *);
  FeX(const std::string& n, TLogLevel, IRootNtupleWriterTool *, IRootNtupleWriterTool *);
  
  int initialize();
  
  //1st arg: target
  //2nd arg: flat image (pixel values)
  int execute(int, std::vector<float>&);
  
  //IIncidentListener impl
  virtual void handle(const Incident&);
  
  void addPreProcessorTool(IPreProcessorTools*);
  
protected:

  bool m_first;
  bool m_stop;

  template <class T>
  float CalcMedian( typename std::vector<T>::const_iterator, typename std::vector<T>::const_iterator);

  IRootNtupleWriterTool * m_ntuple_helper;
  IRootNtupleWriterTool * m_ntuple_helper_metadata;
  
  std::vector<IPreProcessorTools*> m_pre_tools;
  
  template <class T>
  void pushBack(const std::string & , const T*, IRootNtupleWriterTool *);
  
  template <class T>
  void pushBack_meta(const std::string & , const T&, IRootNtupleWriterTool *);
  
  
private:

   void Register();

};
