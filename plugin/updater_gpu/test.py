#pylint: skip-file
import numpy as np
import xgboost as xgb
import os
import pandas as pd
import urllib2

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_last_eval_callback(result):

	def callback(env):
		result.append(env.evaluation_result_list[-1][1])
	
	callback.after_iteration = True
	return callback


def load_adult():
	path = "../../demo/data/adult.data"

	if(not os.path.isfile(path)):
		data = urllib2.urlopen('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data')
		with open(path,'wb') as output:
			  output.write(data.read())
	
	train_set = pd.read_csv( path, header=None)

	train_set.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 
              'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
             'wage_class']
	train_nomissing = train_set.replace(' ?', np.nan).dropna()
	for feature in train_nomissing.columns: # Loop through all columns in the dataframe
	    if train_nomissing[feature].dtype == 'object': # Only apply for columns with categorical strings
		train_nomissing[feature] = pd.Categorical(train_nomissing[feature]).codes # Replace strings with an integer

	y_train = train_nomissing.pop('wage_class')

	return xgb.DMatrix( train_nomissing, label=y_train)


def load_higgs():
	higgs_path = '../../demo/data/training.csv'
	dtrain = np.loadtxt(higgs_path, delimiter=',', skiprows=1, converters={32: lambda x:int(x=='s'.encode('utf-8')) } )

	#dtrain = dtrain[0:200000,:]
	label  = dtrain[:,32]
	data   = dtrain[:,1:31]
	weight = dtrain[:,31] 

	return xgb.DMatrix( data, label=label, missing = -999.0, weight=weight )

def load_dermatology():
	data = np.loadtxt('../../demo/data/dermatology.data', delimiter=',',converters={33: lambda x:int(x == '?'), 34: lambda x:int(x)-1 } )
	sz = data.shape

	X = data[:,0:33]
	Y = data[:, 34]

	return xgb.DMatrix( X, label=Y)

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

#Check GPU test evaluation is approximately equal to CPU test evaluation
def check_result(cpu_result, gpu_result):
	for i in range(len(cpu_result)):
		if not isclose(cpu_result[i], gpu_result[i], 0.1, 0.02):
			return False
	
	return True


#Get data
data = []
params = []
data.append(load_higgs())
params.append({})


data.append( load_adult())
params.append({})

data.append(xgb.DMatrix('../../demo/data/agaricus.txt.test'))
params.append({'objective':'binary:logistic'})

#if(os.path.isfile("../../demo/data/dermatology.data")):
data.append(load_dermatology())
params.append({'objective':'multi:softmax', 'num_class': 6})

num_round = 5

num_pass = 0
num_fail = 0

test_depth = [ 1, 6, 9, 11, 15 ]
#test_depth = [ 1 ]

for test in range(0, len(data)):
	for depth in test_depth:
		xgmat = data[test]
		cpu_result = []
		param = params[test]
		param['max_depth'] = depth
		param['updater'] = 'grow_colmaker'
		xgb.cv(param, xgmat, num_round, verbose_eval=False, nfold=5, callbacks=[get_last_eval_callback(cpu_result)])

		#bst = xgb.train( param, xgmat, 1);
		#bst.dump_model('reference_model.txt','', True)

		gpu_result = []
		param['updater'] = 'grow_gpu'
		xgb.cv(param, xgmat, num_round, verbose_eval=False, nfold=5, callbacks=[get_last_eval_callback(gpu_result)])

		#bst = xgb.train( param, xgmat, 1);
		#bst.dump_model('dump.raw.txt','', True)

		if check_result(cpu_result, gpu_result):
			print(bcolors.OKGREEN + "Pass" + bcolors.ENDC)
			num_pass = num_pass + 1
		else:
			print(bcolors.FAIL + "Fail" + bcolors.ENDC)
			num_fail = num_fail + 1
		
		print("cpu rmse: "+str(cpu_result))
		print("gpu rmse: "+str(gpu_result))

print(str(num_pass)+"/"+str(num_pass + num_fail)+" passed")
