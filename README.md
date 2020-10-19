# Heterogeneous Newton Boosting Machine

The attached code contains:
- A generic Heterogeneous Newton Boosting Machine class, `HNBM`, that selects the base hypothesis class at each boosting iteration from a list of generic scikit-learn regressors.
- A child class, `SnapBoost`, that inherits from `HNBM` and provides a list of `DecisionTreeRegressors` (or varying depth) as well as a `KernelRidgeRegressor`.
- An example of training and evaluating `SnapBoost` for both a synthetic regression task and a synthetic classification task.

This code is **not** the high-performance implementation of `SnapBoost` that was used for the experiments in the manuscript. 
Instead, it is an exemplary implementation with very few dependencies intended to enable reviewers to play around with the algorithms for themselves. 

The optimized code is written in C++ and uses various optimizations, such as histogram statistics for tree-building, that are not present in the scikit-learn classes.
The optimized implementation will soon be made freely available via our organization's public anaconda channels. 
However, we are unable to provide access at this point in the review process without violating the conference's double-blind rules.

## Dependencies 

The attached Python code has the following dependencies:
```
scikit-learn
numpy 
```

## Executing the code

```bash
$ python snapboost.py
SnapBoost RMSE     (test set): 98.9374
SnapBoost log_loss (test set): 0.3861
```
