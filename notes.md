
# Decision Trees
works to perform series of splits to reduce the inpurity
entropy = sum_i{-p_i * log_2{p_i}}
  - information gain = entropy(parent) - [weighted average]entryopy(children)
  - decision tree maximizes information gain

# Regression
Discrete: values with no order
Continuous: values with an ordering

# Unsupervised Learning
Finding structure in data without labels
Clustering
Dimensionality Reduction

## 8 - Clustering
### K-means
#### Pseudo code:
Randomly draw clusters centers then:
  1. Assign
    - assign all points to the closest cluster
  2. Optimize
    - move the cluster center to reduce the sum of the quadratic length to each point in the cluster

Iterate until clusters stop moving very much

#### Limits
Non-deterministic for fixed data and fixed # of clusters (can give different results)
  - highly depends on initial starting points of clusters
  - ex: with 3 clusters possible to group 2 as 1 and split 1 between 2 clusters

# 9 - Feature Scaling
Convert input data into range from [0, 1] where the min point is 0 and the max
point is 1.

```equation
x' = (x-x_{min}) / (x_{max}-x_{min})
```

outliers greatly affect data that has been scaled
  - ex: 100, 250, 900, 1000000 => 0, ~0, ~0, 1

# 10 - Text Learning
**Bag of words:** reduce input space to a set of words.  Input to classifier becomes vector
of word counts.  

Ex: bag: ['hi', 'one', 'a'], input "one one hi" => [1, 2, 0]

**Stemming:** Reduce multiple related words to same "base" word.

Ex: "responsive, responsivity, responsiveness" => "respons"

## TfIdf Representation
**Tf:** term frequency (like bag of words)

**Idf:** inverse document frequency (weighting by how often word occurs in corpus)

Weights rare words higher than common words

## Good Practice
Stem input, then reduce to a bag of words

# 11 - Feature Selection
## Adding a new feature
  1. Use intuition to guess what new feature / info might have patterns
  2. code up this feature (add the feature to the existing data)
  3. visualize to see if the feature added value
  4. repeat

Ex: Enron emails, add "# emails from known persons of interest" as a feature to try and detect other persons of interest
  - no real correlation, but what about "% of emails from knwown POIs?" (intuition)

## Getting Rid of Features
**FEATURES != INFORMATION**

**Feature:** attempt to _access_ information

**Selectors** (% or K-best) can be given training data and judge how good each feature is at predictions.  It then filters out the features that aren't very good

Ex: common words in text classifying convery little information and end up being removed

## Bias Variance Dilemma
few features => likely high bias (doesn't learn as much, high error on training set)

many features => likely high variance (overfits, does not generalize well, high error on training set)

carefully minimized SumSquareErrors => same as many features

**Idea:** Fit classifier/regression to your problem with as little features as necessary

## Regularization

Automatically tune # of features to maximize quality of model.
Ex: 1 feature => low quality

    100 features => low quality

Regularization would automatically find the sweet spot between these two points.

Some algorithms automatically do regularization.

### Regularization in Regression
#### Lasso Regression
Each feature must help more than the arbitrary penalty otherwise that coefficient gets set to zero.

minimize SumSqErr + lambda * |B|
lambda: parameter
B: coeficients of regression

### outliers
Looking for overly "important" features with regularization can sometimes help you find features that are outliers and should not be included.

# 12 - PCA: Principle Component Analysis

translates and rotates the axis of the features to better align with the dimensions of the data.

Ex: diagonal line of data, rotate so diagonal line is now a horizontal line along the x-axis, centre of co-ord system is at centre of diagonal line.  Other axes get orthogonal directions.

## Measureable vs Latent Featrures

**Measureable:** sq footage, # of rooms, school ranking, neighborhood safety
**Latent:** size, neighborhood

How to reduce 4 measurable features to 2 so we get to heart of the info?
  - see selectors like earlier
    - not always best solution

## Principle Components

Preserve information by making a **composite** feature from many features.  This is a **principle component**.

Ex: graph # of rooms and sq footage together, determine a function that approximates their relation as a projection (like a diagonal line that each point can be projected a certain distance to be on the line)

### how to determine
**principal component** is the direction that has the **largest variance** (spread).  This minimizes information loss since no other directions have higher variance.

## PCA as a General Algorithm for Feature Transformation
**Useful for unsupervised learning**

Feed all 4 features into PCA, output is **principal components** ranked by "relative power" of the **principal components**.

Ex: [sq footage, #rooms, school rating, safety rating] becomes

1:[school rating, safety rating], 2:[sq footage, #rooms]

## Summary of PCA
  - systemized way to transform input features into **principal components**
  - use **PC**s as new features
  - **PC**s are directions that maximize variance (minimize information loss) when you project/compress down onto them
  - more variance along a **PC**, higher that **PC** is ranked
    - 1st PC, most info, 2nd PC, 2nd most info, etc
  - **PC**s are orthogonal (do not overlap, they are "independent" features)
  - max # of **PC**s is # of input features

## When to use PCA
  - find latent features driving the data
  - dimensionality reduction
    - visualize high-dimensional data
    - reduce noise
    - improves performance of other algorithms

## PCA for Facial Recognition (Eigenfaces)
Why is it well suited?
  - pictures have high dimensionality
  - faces have patterns that can be captured in smaller # of dimensions