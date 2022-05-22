from sklearn.ensemble import ExtraTreesClassifier
import pandas
import numpy
from sklearn.feature_selection import RFE
from RFE import utils

csv_dir = r'G:\VS\rf_rank\AB\Normal_all.csv'

x_train, x_test, y_train, y_test = utils.load_handle_data(csv_dir)
#Feature extraction
model = ExtraTreesClassifier()
rfe = RFE(model, n_features_to_select=985)
fit = rfe.fit(x_train, y_train)
acc = fit.score(x_test, y_test)
prediction = rfe.predict(x_test)
y = list(y_test)
count_tp = 0
count_tn = 0
count_fn = 0
count_fp = 0
for i in range(0, 5867):
    if prediction[i] == y[i] and y[i] == 1:
        count_tp += 1
    elif prediction[i] == y[i] and y[i] == 0:
        count_tn += 1
    elif prediction[i] != y[i] and prediction[i] == 1:
        count_fp += 1
    else:
        count_fn += 1
print(count_tp, count_tn, count_fp, count_fn)
# print("Num Features: %d"% fit.n_features_)
# print("Selected Features: %s"% fit.support_)
# print("Feature Ranking: %s"% fit.ranking_)
print("Model accuracy: %f"% acc)