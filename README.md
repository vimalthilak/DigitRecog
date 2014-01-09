#Digit Recognizer

This Digit Recognizer package provides an example of the use of supervised (machine) learning methods to classify large datasets. In the present case, the dataset consists in a large set of images representing handwritten digits (0 to 9). The goal is to be able to predict to which class (i.e. which digit) a given instance (image) belongs to. 

This dataset has been studied extensively (see [Yann LeCun's webpage][MNIST]); current best efforts with state of the art classification methods achieve remarkable success rates. The aim of this digit recognizer is not to improve on those results but to provide a testbed for understanding the output of a classifier and making the most out of it. Here, decision trees in the form of random forests will be used since they can readily handle categorical data in a multiclass classifcation problem. 

The process is divided into two steps:

1. A feature extraction algorithm processes all the images and computes a number of observables in order to reduce the complexity of the problem.
2. Using those observables, the random forest classifer is trained and then used to classify the images according to the digits they represent. 

The functionality of the random forest classifier is enhanced by providing _scores_ (or probability estimates) of each digit for every image. Those scores are then calibrated to the true classifier response. Instead of being able to only predict the most likely digit an image represents (e.g.: this image represents a 3), the calibration procedure provides us with the probabilities for an image to represent a given digit (e.g.: 10% chance it's a 0, 1% chance it's a 1, 5% chance it's a 2, 65% chance it's a 3, etc. ).

## Installation

A makefile is provided for unix-like operating systems (tested on Mac OS X 10.7.5). Requirements: 
* C++11 compliant c++ compiler (e.g. g++ 4.8)
* [boost][] library
* Python 2.x (e.g. version 2.7.1)
* [swig][] 2.x 
* [ROOT][] 5.x 
* [OpenCV][] 2.4.6.1

(All the above can be obtained through [macports][] for Max OS X systems.)

* ``simplefwk-services`` package
* ``simplefwk-utilitytoolsinterfaces`` package
* ``simplefwk-utilitytools`` package

## Usage

The input dataset can be obtained from the [kaggle][] website. It consists in 42000 28-by-28 pixels gray-scale images of handwritten digits. Eventhough the modules and algorithms are implemented in C++, most of the steering is done from Python (thanks to swig's automatic interface generation). 

### Feature Extraction

Ideally, one would use the 784 pixel (integer) values of each image as features, leaving to the classifier the task of learning about correlations and building an optimal model of our data in order to yield reliable predictions. A maximal amount of information can then be exploited. An obvious drawback is a lengthy training process. In order to speed up the training of the classifier, a set of 70 potentially discriminating features are systematically computed from each image. 

As a first step, the image is preprocessed:

* A [Gaussian blur][gblur] is applied to smooth out the image (see ``src/GaussianBlurTool.cxx``).   
* A [Canny edge detection][canny] algorithm (see ``src/CannyEdgeTool.cxx``) is used to convert the original image to a binary image consisting of only the edges.

<img src="https://raw.github.com/chapleau/DigitRecog/master/doc/6_plain.png" alt="6 plain" height="350" width="350"> <img src="https://raw.github.com/chapleau/DigitRecog/master/doc/6_processed.png" alt="6 plain" height="350" width="350">

The two images above represent the same instance of a _6_, before (left) and after (right) the preprocessing steps (the extra lines, markers, colors, are just visual aids). The features are computed from the preprocessed images using the pixel positions and includes (see ``src/FeX.cxx``):

* The center of mass (depicted by the black pixel in the image here)
* Axis minimizing the second (inertia) moment (blue axis, red axis is orthogonal)
* Measure of symmetry of the image with respect to the axes defined by the second moments.
* Five number summary (minimum, first quartile, median, third quartile, maximum) of the pixel positions and angle with respect to the inertia axes, computed in each quadrant (the markers identify the different quadrants).

All the features are saved as a TTree in a ROOT file (see ``simplefwk-utilitytools`` package for more details) for easy and efficient future access. The ``run/run.py`` Python file can be used to run the feature extraction algorithm (from the ``run/`` directory, using the Bash shell):

````shell
PYTHONPATH=$PYTHONPATH:`pwd`/../../ python ./run.py
````

### Classification by Random Forests

The classification of the dataset is done using [random forests][rf] using the [OpenCV][] machine learning library. It involves three distinct steps: the cross-validation, training, and testing phases. The whole process can be run using the provided Python steering file (from the ``run/`` directory, using the Bash shell):

````shell
PYTHONPATH=$PYTHONPATH:`pwd`/../../ python ./run_cl.py
````

#### Probability estimation trees

In many implementations of the random forests classifier (such as the one present in the OpenCV library), the _forest_ consists in an ensemble of classification trees. Each leaf in such trees is associated with a single class label that is determined by a vote during the training phase. Modifications were made to the OpenCV impementation (see ``src/rtrees.hpp``) in order to operate on an ensemble of probability estimation trees. In this case, the relative class frequency in a leaf that is obtained during the training phase is used as an estimation of the class membership probability. The estimated class probablity for the forest is taken as the average, over the whole ensemble, of the single tree relative class frequency (see [this paper][bostrom07]).

Those probability estimates will be denoted as _scores_. If two events are classified as members of a class _c_, the one with the highest score is more likely to be a true member of class _c_. In order to be able to interpret those scores as the chance of membership of a class, calibration is necessary. The calibration maps scores to empirical class membership probabilities. Accurate class probability estimates are necessary when combining the output of the classifier with other independent sources of information or with different classifiers. (see [this paper][kdd2002]). 

The calibration is discussed in more details in the cross-validation section.


#### Cross-validation & training

A _k_-fold cross-validation (with _k_ set to 4 for practical purposes) is performed on half of the dataset. It consists in randomly partitioning the sample into _k_ subsamples of roughly equal size, leaving out one for validation/testing purposes and using the remaining subsamples as training data. The procedure is repeated _k_ times, each time using a different subsample for testing, and the results are combined. Each event can then be tested using a classifier trained on an independent set of events. 

Each instance of the cross-validation procedure is executed in parallel in a unique thread. The original OpenCV implementation was modified in order to (notably) monitor the progress of the computations:
<img src="https://raw.github.com/chapleau/DigitRecog/master/doc/cv_terminal.png" alt="cv">

The main purpose of cross-validation is to find out the best classifier parameter values to be used. This can be done by scanning the parameter space for the set of values that allows the classifier to perform at its best. A good set of parameter values were found that way and are used in ``run/run_cl.py``. 

The calibration functions (i.e. mapping from scores to probabilities estimates) is determined during cross-validation. Ideally, when dealing with a multiclass classifier, a non-trivial multidimensional mapping function would need to be determined. Here, for simplicity, a calibration is determined for each class individually. The calibrated probability estimates are then normalized so that the sum to 1.


[MNIST]: http://yann.lecun.com/exdb/mnist/
[boost]: http://www.boost.org/
[macports]: http://www.macports.org/
[ROOT]: http://root.cern.ch
[swig]: http://swig.org
[OpenCV]: http://opencv.org/
[kaggle]: http://www.kaggle.com/c/digit-recognizer/data
[gblur]: http://en.wikipedia.org/wiki/Gaussian_blur
[canny]: http://en.wikipedia.org/wiki/Canny_edge_detector
[rf]: http://www.stat.berkeley.edu/users/breiman/RandomForests/cc_home.htm
[bostrom07]: http://people.dsv.su.se/~henke/papers/bostrom07c.pdf
[kdd2002]: http://www.research.ibm.com/people/z/zadrozny/kdd2002-Transf.pdf
