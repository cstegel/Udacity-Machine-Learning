
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
  - ex with 3 clusters possible to group 2 as 1 and split 1 between 2 clusters

# 9 - Feature Scaling
Convert input data into range from [0, 1] where the min point is 0 and the max
point is 1.

```equation
x' = (x-x_{min}) / (x_{max}-x_{min})
```