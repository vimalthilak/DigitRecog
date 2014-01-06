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

The input dataset can be obtained from the [kaggle][] website. It consists in 42000 28-by-28 pixels gray-scale images of handwritten digits. Eventhough the modules and algorithms are implemented in C++, all the steering is done from Python (thanks to swig's automatic interface generation). 

### Feature Extraction

Ideally, one would use the 784 pixel (integer) values of each image as features, leaving to the classifier the task to learn correlations among those in order to be able to yield predictions. A maximal amount of information can then be exploited. An obvious drawback is a lengthy training process. In order to speed up the training of the classifier, a set of 70 potentially discriminating features are systematically computed from each image. 

As a first step, the image is preprocessed:

* A [Gaussian blur][gblur] is applied to smooth out the image (see ``src/GaussianBlurTool.cxx``).   
* A [Canny edge detection][canny] algorithm (see ``src/CannyEdgeTool.cxx``) is used to convert the original image to a binary image consisting of only the edges.


[MNIST]: http://yann.lecun.com/exdb/mnist/
[boost]: http://www.boost.org/
[macports]: http://www.macports.org/
[ROOT]: http://root.cern.ch
[swig]: http://swig.org
[OpenCV]: http://opencv.org/
[kaggle]: http://www.kaggle.com/c/digit-recognizer/data
[gblur]: http://en.wikipedia.org/wiki/Gaussian_blur
[canny]: http://en.wikipedia.org/wiki/Canny_edge_detector
