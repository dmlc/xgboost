import xgboost as xgb
import numpy as np

clf = xgb.XGBClassifier()
clf.fit(np.array([[1,2],[3,4]]), np.array([0,1]))
clf.save_model('model.json')
