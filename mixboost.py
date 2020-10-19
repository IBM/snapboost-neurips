import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.base import clone

class MeanSquaredError:
    ''' Mean squared error loss function
    '''
    @staticmethod
    def compute_derivatives(y, f):
        g = 2*(f - y)
        h = 2.0 * np.ones(y.shape[0])
        return g, h

class LogisticLoss:
    ''' Logistic loss function
    '''
    @staticmethod
    def compute_derivatives(y, f):
        tmp = np.exp(-np.multiply(y, f))
        tmp2 = np.divide(tmp, 1+tmp)
        g = -np.multiply(y, tmp2)
        h = np.multiply(tmp2, 1.0-tmp2)
        return g, h

class HNBM:
    ''' A generic Heterogeneous Newton Boosting Machine

        Args:
            loss (class): loss function
            num_iterations (int): number of boosting iterations
            learning_rate (float): learning rate
            base_learners (list): list of base learners
            probabilities (list): list of sampling probabilities

        Attributes:
            ensemble_ (list): Ensemble after training
    '''
    def __init__(self, loss, num_iterations, learning_rate, base_learners, probabilities):
        self.loss_ = loss
        self.num_iterations_ = num_iterations
        self.learning_rate_ = learning_rate
        self.base_learners_ = base_learners
        self.probabilities_ = probabilities
        self.ensemble_ = []

    def fit(self, X, y):
        ''' Train the model

        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Labels

        '''
        z = np.zeros(X.shape[0])
        self.ensemble_ = []
        for i in range(0, self.num_iterations_):
            g, h = self.loss_.compute_derivatives(y, z)
            base_learner = clone(np.random.choice(self.base_learners_, p=self.probabilities_))
            base_learner.fit(X, -np.divide(g, h), sample_weight=h)
            z += base_learner.predict(X) * self.learning_rate_
            self.ensemble_.append(base_learner)

    def predict(self, X):
        ''' Predict using the model

        Args:
            X (np.ndarray): Feature matrix

        '''
        preds = np.zeros(X.shape[0])
        for learner in self.ensemble_:
            preds +=  self.learning_rate_ * learner.predict(X)
        return preds

class MixBoost(HNBM):
    ''' A particular realization of a HNBM that uses decision trees
        and kernel ridge regressors

        Args:
            loss (class): loss function
            num_iterations (int): number of boosting iterations
            learning_rate (float): learning rate
            p_tree (float): probability of selecting a tree at each iteration
            min_max_depth (int): minimum maximum depth of a tree in the ensemble
            max_max_depth (int): maximum maximum depth of a tree in the ensemble
            alpha (float): L2-regularization penalty in the ridge regression
            gamma (float): RBF-kernel parameter

    '''

    def __init__(self, loss=MeanSquaredError, num_iterations=100, learning_rate=0.1, p_tree=0.8, 
                       min_max_depth=4, max_max_depth=8, alpha=1.0, gamma=1.0):

        base_learners = []
        probabilities = []

        # Insert decision tree base learners
        depth_range = range(min_max_depth,  1+max_max_depth)
        for d in depth_range:
            base_learners.append(DecisionTreeRegressor(max_depth=d, random_state=42))
            probabilities.append(p_tree/len(depth_range))

        # Insert kernel ridge base learner
        base_learners.append(KernelRidge(alpha=alpha, kernel='rbf', gamma=gamma))
        probabilities.append(1.0-p_tree)

        super().__init__(loss, num_iterations, learning_rate, base_learners, probabilities)

def test(classification=False):
    ''' Test MixBoost on a synthetic learning task:
        
        Args:
            classification (bool): generate a classification task (if True) 
                                   or a regression task (if False)
    '''

    # for deterministic results across runs
    np.random.seed(42)

    # construct a MixBoost object
    model = MixBoost(loss=LogisticLoss if classification else MeanSquaredError)

    # generate a dataset (regression task)
    if classification:
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        # we assume [-1,+1] labels
        y = 2*y-1
    else:
        X, y = make_regression(n_samples=1000, n_features=20, random_state=42)

    # split into train/test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # train a MixBoost model
    model.fit(X_train, y_train)

    # predict using the MixBoost model
    preds = model.predict(X_test)

    # evaluate the model
    if classification:
        logloss= log_loss(y_test, 1.0/(1.0+np.exp(-preds)))
        print("MixBoost log_loss (test set): %.4f" % (logloss)) 
    else:
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        print("MixBoost RMSE     (test set): %.4f" % (rmse))


if __name__ == "__main__":
    test(classification=False)
    test(classification=True)

