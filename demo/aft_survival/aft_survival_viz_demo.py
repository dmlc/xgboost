"""
Visual demo for survival analysis (regression) with Accelerated Failure Time (AFT) model.

This demo uses 1D toy data and visualizes how XGBoost fits a tree ensemble. The ensemble model
starts out as a flat line and evolves into a step function in order to account for all ranged
labels.
"""
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 13})

# Function to visualize censored labels
def plot_censored_labels(X, y_lower, y_upper):
    def replace_inf(x, target_value):
        x[np.isinf(x)] = target_value
        return x
    plt.plot(X, y_lower, 'o', label='y_lower', color='blue')
    plt.plot(X, y_upper, 'o', label='y_upper', color='fuchsia')
    plt.vlines(X, ymin=replace_inf(y_lower, 0.01), ymax=replace_inf(y_upper, 1000),
               label='Range for y', color='gray')

# Toy data
X = np.array([1, 2, 3, 4, 5]).reshape((-1, 1))
INF = np.inf
y_lower = np.array([ 10,  15, -INF, 30, 100])
y_upper = np.array([INF, INF,   20, 50, INF])

# Visualize toy data
plt.figure(figsize=(5, 4))
plot_censored_labels(X, y_lower, y_upper)
plt.ylim((6, 200))
plt.legend(loc='lower right')
plt.title('Toy data')
plt.xlabel('Input feature')
plt.ylabel('Label')
plt.yscale('log')
plt.tight_layout()
plt.show(block=True)

# Will be used to visualize XGBoost model
grid_pts = np.linspace(0.8, 5.2, 1000).reshape((-1, 1))

# Train AFT model using XGBoost
dmat = xgb.DMatrix(X)
dmat.set_float_info('label_lower_bound', y_lower)
dmat.set_float_info('label_upper_bound', y_upper)
params = {'max_depth': 3, 'objective':'survival:aft', 'min_child_weight': 0}

accuracy_history = []
def plot_intermediate_model_callback(env):
    """Custom callback to plot intermediate models"""
    # Compute y_pred = prediction using the intermediate model, at current boosting iteration
    y_pred = env.model.predict(dmat)
    # "Accuracy" = the number of data points whose ranged label (y_lower, y_upper) includes
    #              the corresponding predicted label (y_pred)
    acc = np.sum(np.logical_and(y_pred >= y_lower, y_pred <= y_upper)/len(X) * 100)
    accuracy_history.append(acc)
    
    # Plot ranged labels as well as predictions by the model
    plt.subplot(5, 3, env.iteration + 1)
    plot_censored_labels(X, y_lower, y_upper)
    y_pred_grid_pts = env.model.predict(xgb.DMatrix(grid_pts))
    plt.plot(grid_pts, y_pred_grid_pts, 'r-', label='XGBoost AFT model', linewidth=4)
    plt.title('Iteration {}'.format(env.iteration), x=0.5, y=0.8)
    plt.xlim((0.8, 5.2))
    plt.ylim((1 if np.min(y_pred) < 6 else 6, 200))
    plt.yscale('log')

res = {}
plt.figure(figsize=(12,13))
bst = xgb.train(params, dmat, 15, [(dmat, 'train')], evals_result=res,
                callbacks=[plot_intermediate_model_callback])
plt.tight_layout()
plt.legend(loc='lower center', ncol=4,
           bbox_to_anchor=(0.5, 0),
           bbox_transform=plt.gcf().transFigure)
plt.tight_layout()

# Plot negative log likelihood over boosting iterations
plt.figure(figsize=(8,3))
plt.subplot(1, 2, 1)
plt.plot(res['train']['aft-nloglik'], 'b-o', label='aft-nloglik')
plt.xlabel('# Boosting Iterations')
plt.legend(loc='best')

# Plot "accuracy" over boosting iterations
# "Accuracy" = the number of data points whose ranged label (y_lower, y_upper) includes
#              the corresponding predicted label (y_pred)
plt.subplot(1, 2, 2)
plt.plot(accuracy_history, 'r-o', label='Accuracy (%)')
plt.xlabel('# Boosting Iterations')
plt.legend(loc='best')
plt.tight_layout()

plt.show()
