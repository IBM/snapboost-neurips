# SnapBoost Sample Code

The attached code contains:
- A generic Heterogeneous Newton Boosting Machine class, `HNBM`, that selects the base hypothesis class at each boosting iteration from a list of generic scikit-learn regressors.
- A child class, `SnapBoost`, that inherits from `HNBM` and provides a list of `DecisionTreeRegressors` (or varying depth) as well as a `KernelRidgeRegressor`.
- An example of training and evaluating `SnapBoost` for both a synthetic regression task and a synthetic classification task.

This code is **not** the high-performance implementation of `SnapBoost` that was used for the experiments in the manuscript. 
Instead, it is an exemplary implementation with very few dependencies intended to enable readers to play around with the algorithms for themselves. 

The optimized code is written in C++ and uses various optimizations, such as histogram statistics for tree-building, that are not present in the scikit-learn classes.
We will soon update this repo with instructions for how to install a package containing the optimized SnapBoost.

## Executing the code

```bash
$ pip install -r requirements.txt
$ python snapboost.py
SnapBoost RMSE     (test set): 98.9374
SnapBoost log_loss (test set): 0.3861
```
