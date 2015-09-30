# Overview

The final project required me to train a Person Of Interest classifier for people who worked at enron given a bunch of features about the emails that each person sent.  The aim was to have high accuracy and precision, although the instructions have changed a bit since I first started this (they no longer have a set number to obtain).

My attempt to solve this was to algorithmically try using as many different types of classifiers using their default arguments.  I then perform dimensionality reduction on the data set features to have data sets with everything from 3 to "n" features, where "n" was the original number of features given (I think it was around 18 or so).  Each dataset gets split into test/training data according to the methods provided by Udacity.  Each classifier is then trained and tested on each dimensionaly reduced dataset with the evaluation statistics (accuracy, f1, f2, etc) saved.  

This process takes a long time since it is done sequentially (I think it took about an hour or so).  The saved stats are persisted in a pickle file after they have run once and are only re-calculated if I change what parameters I want the classifiers to run with.

I ended up finding a couple classifier/dataset combinations that satisfied the precision and recall that was needed.  The highest f2 score I got wass 0.58 with precision 0.28 and recall 0.78.  It seemed like this method elicited a lot of variations between high recall, low precision and low recall, high precision.
