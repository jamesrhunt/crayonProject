from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def getClassAndHyp(models_=None):
    # Define the list of classifiers to evaluate

    classifiers_ = [
        LogisticRegression(max_iter=500),
        KNeighborsClassifier(),
        RandomForestClassifier(),
        SVC(probability=True)
    ]

    ########################################################################

    # HYPER PARAMETER TUNING

    hyperparameters_ = [
        {
            'classification__C': [0.01, 0.1, 1, 10, 50, 100], # regularisation parameter: larger c = more overfitting
            'classification__penalty': ['l1', 'l2'], # l1: lasso, l2: ridge
            'classification__solver': ['newton-cg', 'lbfgs'] #trying two solvers because lbfgs doesn't work for l1 penalty (lasso)
        },
        {
            'classification__n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19],
            'classification__weights': ['uniform', 'distance'], # distance to take into account distance of neighbours
            'classification__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'] # auto checks between ball (sparse data) 
                                                                                   # and kd (dense data), brute compares all points!
        },
        {
            'classification__n_estimators': [30, 40,100, 200], # number of decision trees
            'classification__criterion': ['gini', 'entropy'], # function used to measure quality of split
            'classification__max_depth': [None, 5, 10, 20, 40, 80] # max depth for each tree
        },
        {
            'classification__C': [0.01, 0.1, 1, 10], # regularisation based on misclassification and margin width
            'classification__kernel': ['linear', 'rbf'] # kernel used to transform, rbf: guassian better for non-linear
        }
    ]

        # Filter the classifiers and hyperparameters based on the provided classes and hyperparameters
    if models_ is not None:
        classifiers_ = [classifiers_[i] for i in models_]
        hyperparameters_ = [hyperparameters_[i] for i in models_]

    print("Running the following classifiers:")
    print(models_, classifiers_)
    print("With these hyperparameters:")
    print(models_, hyperparameters_)

    
    return classifiers_, hyperparameters_