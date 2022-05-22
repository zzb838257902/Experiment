import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from RFECV_process import utils


def RFECV_svm(file_dir):


    x_train, y_train = utils.load_handle_data(file_dir)
    # Create the RFE object and compute a cross-validated score.
    rf = RandomForestClassifier(n_estimators=600, min_samples_leaf=5, n_jobs=-1, random_state=None)
    # The "accuracy" scoring is proportional to the number of correct
    # classifications
    rfecv = RFECV(estimator=rf, step=1, cv=StratifiedKFold(10),
                  scoring='accuracy')
    rfecv.fit(x_train, y_train)

    print("Optimal number of features : %d" % rfecv.n_features_)
    print("Ranking of features : %s" % rfecv.ranking_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (rf)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()

if __name__ == '__main__':
    RFECV_svm(file_dir)