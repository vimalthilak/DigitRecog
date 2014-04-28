#
# $Id$
#
# File: D3PDMakerTest/python/difftuple.py
# Author: snyder@bnl.gov
# Date: Feb, 2010
# Purpose: Diff two root tuple files.
#

# Always run in batch mode.
import os
if os.environ.has_key('DISPLAY'):
    del os.environ['DISPLAY']

import ROOT
import PyCintex
import types
import os
from fnmatch import fnmatch

# new : old
renames = {
    }

# List of branches to ignore for various versions.
# Format is a list of tuples: (V, V, V, [BRANCHES...])
# If any of the strings in V appear in CMTPATH, then
# we do not ignore the branches.  Otherwise, we ignore
# branches matching any of the glob paths in BRANCHES.
_ignore_branches = [
    ]

# Should a branch be ignored?
def ignore_p (br):
    ver = os.environ.get ('CMTPATH')

    for vlist in _ignore_branches:
        for v in vlist[:-1]:
            found = False
            if ver and ver.find (v) >= 0:
                found = True
                break
        if found: continue
        for b in vlist[-1]:
            if fnmatch (br, b):
                return True
    return False
        


def topy (o):
    if type(o).__name__.startswith ('vector<'):
        ll = list(o)
        if ll and type(ll[0]).__name__.startswith ('vector<'):
            ll = [list(l) for l in ll]
        return ll
    return o


inttypes = [types.IntType, types.LongType]
def compare (o1, o2, thresh = 1e-6, ithresh = None):
    # Allow comparing int/long int.
    if type(o1) in inttypes and type(o2) in inttypes:
        if o1 < 0: o1 = o1 + (1<<32)
        if o2 < 0: o2 = o2 + (1<<32)
        return o1 == o2
    if type(o1) != type(o2):
        #if type(o1) == type([]) and len(o1) == 1 and type(o1[0]) == type(o2):
        #   if not compare(o1[0], o2, thresh=thresh, ithresh=ithresh): return False
        #   else: return True
        #else: 
           return False
    if type(o1) == type([]):
        if len(o1) != len(o2):
            return False
        for i in range(len(o1)):
            if not compare (o1[i], o2[i],
                            thresh=thresh, ithresh=ithresh): return False
        return True
    if type(o1).__name__ in ['map<string,int>',
                             'map<string,float>',
                             'map<string,string>']:
        return ROOT.D3PDTest.MapDumper.equal (o1, o2)
    if type(o1) == type(1.1):
        if ithresh and abs(o1) < ithresh and abs(o2) < ithresh:
            return True
        num = o1-o2
        den = abs(o1)+abs(o2)
        if den == 0: return True
        x = abs(num / den)
        if callable(thresh): thresh = thresh(den)
        if x > thresh:
            print 'fmismatch', o1, o2, x, thresh
            return False
        return True
    return o1 == o2


def mc_eta_thresh (x):
    if x > 36: return 1e-2
    if x > 34: return 2e-3
    if x > 32: return 2e-4
    if x > 30: return 1e-4
    if x > 28: return 1e-5
    return 1e-6


def diff_trees (t1, t2):
    n1 = t1.GetEntries()
    n2 = t2.GetEntries()
    if n1 != n2:
        print 'Different nentries for tree', t1.GetName(), ': ', n1, n2
        n1 = min(n1, n2)
    ##
    #n1 = min(n1, 10)
    print "in diffTrees: ", t1.GetName(),"  ", n1
    ##
    b1 = [b.GetName() for b in t1.GetListOfBranches()]
    b2 = [b.GetName() for b in t2.GetListOfBranches()]
    b1.sort()
    b2.sort()
    branchmap = renames.copy()
    for b in b1:
        if b not in b2:
            bb = branchmap.get (b)
            if not bb or bb not in b2:
                if not ignore_p(b):
                    print 'Branch', b, 'in first tree but not in second.'
                if bb: del branchmap[b]
            else:
                b2.remove (bb)
        else:
            b2.remove (b)
            branchmap[b] = b
    for b in b2:
        if not ignore_p(b):
            print 'Branch', b, 'in second tree but not in first.'

    '''
    from random import sample
    myrange = sample(xrange(n1), 1000)
    for i in myrange:
        #print "chose i: ", i

        t1.GetEntry(i)
        eventnumber = t1.EventNumber

        t2.SetBranchStatus("*", 0)
        t2.SetBranchStatus("EventNumber",1)
        for j in range(n2):
          t2.GetEntry(j)
          if eventnumber == t2.EventNumber:
            #print "got event: ", eventnumber
            t2.SetBranchStatus("*", 1)
            t2.GetEntry(j)
            break
        
        if eventnumber != t2.EventNumber:
           print "Couldn't find event !!!"
           continue
   
    '''
    for i in range (n1):
        t1.GetEntry(i)
        t2.GetEntry(i)
        for b in b1:
            if ignore_p(b): continue
            bb = branchmap.get(b)
            if not bb: continue
            o1 = topy (getattr(t1, b))
            o2 = topy (getattr(t2, bb))

            ithresh = None
            thresh = 1e-6
            if b.find('jet_')>=0 and b.endswith ('_m'): ithresh = 0.1
            if b == 'mc_m': ithresh = 0.1
            if b.find ('_rawcl_etas') >=0: thresh = 2e-4
            if b.endswith ('_convIP'): thresh = 3e-5
            if b.endswith ('_emscale_E'): thresh = 9e-5
            if b.endswith ('_emscale_eta'): thresh = 9e-5
            if b.endswith ('_emscale_m'): ithresh = 0.1
            if b.endswith ('_constscale_E'): thresh = 9e-5
            if b.endswith ('_constscale_eta'): thresh = 9e-5
            if b == 'mc_eta': thresh = mc_eta_thresh
            if b.endswith ('_seg_locX'): ithresh = 2e-12
            if b.endswith ('_seg_locY'): ithresh = 2e-12
            if b == 'MET_Goodness_DeltaEt_JetAlgs_Jet': ithresh = 2e-11
            if b == 'MET_Goodness_EEM_Jet': thresh = 2e-5
            if b == 'MET_Goodness_HECf_Jet': thresh = 3e-6
            if b.find ('_blayerPrediction') >= 0: thresh = 4e-4
            if b.endswith ('_Dip12'): ithresh = 1e-12
            if not compare (o1, o2, thresh = thresh, ithresh = ithresh):
                print 'Branch mismatch', b, 'entry', i, ':', ithresh
                print o1
                print o2

    return


def diff_objstrings (k, s1, s2):
    # nb. != not working correctly for TObjString in 5.26.00c_python2.6
    if not (s1 == s2):
        print 'Objstring', k, 'mismatch:'
        print repr(s1)
        print repr(s2)
    return


def diff_dirs (f1, f2):
    k1 = [k.GetName() for k in f1.GetListOfKeys()]
    k2 = [k.GetName() for k in f2.GetListOfKeys()]
    k1.sort()
    k2.sort()
    if k1 != k2:
        print "Key list mismatch for", f1.GetName(), f2.GetName(), ":"
        print k1
        print k2
    for k in k1:
        #k_1 = k
        #k_2 = k
        #if k != "tt":
        #   k_2 = "tt"
        if k not in k2: continue
        #print 'will compare k_1: ', k_1, '  with k_2: ', k_2
        o1 = f1.Get(k)
        o2 = f2.Get(k)
        if type(o1) != type(o2):
            print 'Type mismatch for ', k
            print o1, o2
        if k == 'Schema':
            pass
        elif isinstance (o1, ROOT.TTree):
            diff_trees (o1, o2)
        elif isinstance (o1, ROOT.TDirectory):
            diff_dirs (o1, o2)
        elif isinstance (o1, ROOT.TObjString):
            diff_objstrings (k, o1, o2)
        else:
            print "skipping:", k, type(o1)
    return


if __name__ == '__main__':
    import sys
    f1 = ROOT.TFile (sys.argv[1])
    f2 = ROOT.TFile (sys.argv[2])
    diff_dirs (f1, f2)
