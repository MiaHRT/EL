import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import shapiro, mannwhitneyu
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, StackingRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.metrics import r2_score
# plt.rc('font',family='DengXian') # 设置plt显示的中文字体

'''load data'''
var_name = {i:"v"+str(i) for i in range(32)}
df_train = pd.read_excel("./data/回归预测.xlsx", sheet_name="训练集", header=None)
df_test = pd.read_excel("./data/回归预测.xlsx", sheet_name="测试集", header=None)
df_train.rename(columns=var_name, inplace=True)
df_test.rename(columns=var_name, inplace=True)
cat_name = {"奥氮平": "c1", "利培酮": "c2", "齐拉西酮": "c3", "阿立哌唑": "c4", "喹硫平": "c5", "氟哌啶醇": "c6", "奋乃静": "c7"}
for i in range(len(df_train)):
    df_train.iloc[i,30] = cat_name[df_train.iloc[i,30]]
for i in range(len(df_test)):
    df_test.iloc[i,30] = cat_name[df_test.iloc[i,30]]

'''view data'''
# print(df_train.head(5))
# df_train.info()
# print(df_train.describe())

'''find missing data'''
# print(df_train.isnull().sum())
# print(df_test.isnull().sum())

'''plot data'''
# cols = df_train.columns.tolist()
# cols.remove("v30")
# cols.remove("v31")
# fig, ax = plt.subplots(figsize=(12, 10), tight_layout=True)
# df_train.hist(column=cols, ax=ax)
# plt.savefig("./image/hist.png")

# tmp = df_train["v30"].value_counts()
# fig, ax = plt.subplots(figsize=(7, 5))
# ax.bar(x=list(tmp.index), height=tmp.tolist())
# ax.set_title("v30")
# plt.grid()
# plt.savefig("./image/bar.png")

# fig, ax = plt.subplots(figsize=(7, 5))
# df_train.hist(column="v31", bins=10, ax=ax)
# plt.savefig("./image/hist_dependent.png")

'''one-hot'''
onehot_tran = pd.get_dummies(df_train["v30"], prefix=None)
df_train = df_train.join(onehot_tran)
df_train.rename(columns=cat_name, inplace=True)
del df_train["v30"]
# df_train.info()

onehot_tran = pd.get_dummies(df_test["v30"], prefix=None)
df_test = df_test.join(onehot_tran)
df_test.rename(columns=cat_name, inplace=True)
del df_test["v30"]
del df_test["c7"]
# df_test.info()

'''Wilcoxon test'''
# tmp = list(cat_name.values())
# for i in range(len(tmp)-1):
#     for j in range(i+1, len(tmp)):
#         test_stat = mannwhitneyu(df_train.loc[df_train[tmp[i]]==1,"v31"], df_train.loc[df_train[tmp[j]]==1,"v31"])
#         print("{}-{}: {}".format(tmp[i],tmp[j],test_stat))
# for i in range(len(tmp)):
#     test_stat = mannwhitneyu(df_train.loc[df_train[tmp[i]]==1,"v31"], df_train.loc[df_train[tmp[i]]==0,"v31"])
#     print("{}: {}".format(tmp[i],test_stat))
del df_train["c7"]

'''Multi-co-linearity'''
# tmp = ["v"+str(i) for i in range(30)]
# cov_mat = df_train[tmp].corr()
# cond_value = np.linalg.cond(cov_mat)
# print("condition number = {}".format(cond_value))
# plt.imshow(cov_mat)
# plt.colorbar()
# plt.title("correlation coefficient")
# plt.show()

'''cross validation'''
def CrossValid(model, X, y):
    kf = KFold(n_splits=10)
    score, n = 0, 0
    for train_index, test_index in kf.split(X, y):
        instance = clone(model)
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        instance.fit(X_train.values, y_train)
        tmp = instance.score(X_test.values, y_test)
        score += tmp
        n += 1
        # print('{}'.format(n), tmp)
    score /= n
    print('Average Score of Model{}'.format(str(model)), score)
    return score

'''Variable Selection'''
'''Lasso regression and Box-Cox transformation (no interaction effect)'''
X = df_train.drop(['v31'], axis=1)
y = df_train['v31']
def box_cox(y, lam):
    if lam==0:
        return np.log(y)
    elif lam>0:
        return (np.power(y,lam)-1)/lam
    else:
        return (1/np.power(y,-lam)-1)/lam
# selecting hyperparameter
# L0 = np.round(np.arange(-1,1.01,0.1),2)
# L1 = np.round(np.arange(0.01,0.21,0.01),2)
# scores = np.zeros((len(L0),len(L1)))
# for i in range(len(L0)):
#     for j in range(len(L1)):
#         tmp = box_cox(y, L0[i])
#         lasso=Lasso(alpha=L1[j], max_iter=10000)
#         scores[i,j] = CrossValid(lasso, X, tmp)
# i,j = np.where(scores==np.max(scores))
# print(L0[i[0]],L1[j[0]],np.max(scores))
# plt.imshow(scores)
# plt.colorbar()
# plt.title(r"$R^2$ (10-fold cross validation)")
# plt.ylabel(r"$\lambda_0$")
# plt.xlabel(r"$\lambda_1$")
# plt.yticks(range(len(L0)),L0,rotation=45)
# plt.xticks(range(len(L1)),L1,rotation=45)
# plt.show()

y = box_cox(y, 0.4)
lasso=Lasso(alpha=0.18, max_iter=10000)
lasso.fit(X, y)
score0 = lasso.score(X, y)
print("lasso score0 = {}".format(score0))
drop_list = list(X.columns[lasso.coef_==0])
# plt.bar(list(X.columns), lasso.coef_)
# plt.ylabel("coef (lasso)")
# plt.grid()
# plt.show()

'''Lasso regression (interaction effect)'''
add_list = []
X = df_train.drop(['v31']+drop_list, axis=1)
for i in range(30):
    for j in range(1,7):
        Xtmp = X.copy()
        Xtmp["v{}-c{}".format(i,j)] = df_train["v{}".format(i)]*df_train["c{}".format(j)]
        lasso=Lasso(alpha=0.18, max_iter=10000)
        lasso.fit(Xtmp, y)
        score = lasso.score(Xtmp, y)
        if lasso.coef_[-1]!=0:
            add_list.append((i,j))
# print(add_list)
for i,j in add_list:
    X["v{}-c{}".format(i,j)] = df_train["v{}".format(i)]*df_train["c{}".format(j)]
y = df_train['v31']
# selecting hyperparameter
# L0 = np.round(np.arange(-1,1.01,0.1),2)
# L1 = np.round(np.arange(0.01,0.21,0.01),2)
# scores = np.zeros((len(L0),len(L1)))
# for i in range(len(L0)):
#     for j in range(len(L1)):
#         tmp = box_cox(y, L0[i])
#         lasso=Lasso(alpha=L1[j], max_iter=10000)
#         scores[i,j] = CrossValid(lasso, X, tmp)
# i,j = np.where(scores==np.max(scores))
# print(L0[i[0]],L1[j[0]],np.max(scores))
# plt.imshow(scores)
# plt.colorbar()
# plt.title(r"$R^2$ (10-fold cross validation)")
# plt.ylabel(r"$\lambda_0$")
# plt.xlabel(r"$\lambda_1$")
# plt.yticks(range(len(L0)),L0,rotation=45)
# plt.xticks(range(len(L1)),L1,rotation=45)
# plt.show()

y = box_cox(y, 0.3)
lasso=Lasso(alpha=0.06, max_iter=10000)
lasso.fit(X, y)
print("lasso score = {}".format(lasso.score(X, y)))
drop_list_interaction = list(X.columns[lasso.coef_==0])
# plt.bar(list(X.columns), lasso.coef_)
# plt.ylabel("coef (lasso)")
# plt.xticks(rotation=45)
# plt.grid()
# plt.show()

df_train_reg = df_train.copy()
df_test_reg = df_test.copy()
for i,j in add_list:
    tmp = "v{}-c{}".format(i,j)
    if tmp not in drop_list_interaction:
        df_train_reg[tmp] = df_train["v{}".format(i)]*df_train["c{}".format(j)]
        df_test_reg[tmp] = df_test["v{}".format(i)]*df_test["c{}".format(j)]
df_train_reg = df_train_reg.drop(drop_list, axis=1)
# df_train_reg.info()
df_test_reg = df_test_reg.drop(drop_list, axis=1)
# df_test_reg.info()

'''Model fitting'''
X_train = df_train.drop(['v31'], axis=1)
y_train = df_train['v31']
X_reg_train = df_train_reg.drop(['v31'], axis=1)
y_reg_train = box_cox(y_train, 0.3)
def inverse_boxcox(y):
    return np.power(0.3*y+1, 10/3)

X_test = df_test.drop(['v31'], axis=1)
y_test = df_test['v31']
X_reg_test = df_test_reg.drop(['v31'], axis=1)

err_models = {}
R2_models = {}

'''Linear Regression'''
lm = LinearRegression()
lm.fit(X_reg_train, y_reg_train)
lm_predict = inverse_boxcox(lm.predict(X_reg_test))
R2_models["Linear Regression"] = r2_score(y_test, lm_predict)
err = lm_predict-y_test
err_models["Linear Regression"] = (np.mean(err), np.var(err))

# ploting coefs
# coef = lm.coef_
# sorted_idx = np.argsort(coef)[::-1]
# plt.bar(np.array((X_reg_train.columns))[sorted_idx], coef[sorted_idx])
# plt.ylabel("coef (lm)")
# plt.xticks(rotation=45)
# plt.grid()
# plt.show()

# ablation experiment
# lm.fit(X_train, y_train)
# print("Linear Regression score = {}".format(lm.score(X_test, y_test)))

'''MLP'''
# selecting hyperparameter
# scores = []
# L = range(5, 51, 5)
# for l in L:
#     mlp = MLPRegressor(hidden_layer_sizes=(l),  activation='logistic', solver='adam', max_iter=10000, shuffle=True,random_state=0)
#     scores.append(CrossValid(mlp, X_reg_train, y_reg_train))
# plt.plot(L, scores)
# plt.xticks(L)
# plt.ylabel(r"$R^2$ (10-fold cross validation)")
# plt.xlabel("hidden layer sizes")
# plt.grid()
# plt.show()

# fit the model
mlp = MLPRegressor(hidden_layer_sizes=(30),  activation='logistic', solver='adam', max_iter=10000, shuffle=True,random_state=0)
mlp.fit(X_reg_train, y_reg_train)
mlp_predict = inverse_boxcox(mlp.predict(X_reg_test))
R2_models["MLP"] = r2_score(y_test, mlp_predict)
err = mlp_predict-y_test
err_models["MLP"] = (np.mean(err), np.var(err))

# ablation experiment
# mlp.fit(X_train, y_train)
# print("mlp score = {}".format(mlp.score(X_test, y_test)))

'''Decision Tree Regressor'''
# selecting hyperparameter
# scores = []
# L = range(3, 11)
# for l in L:
#     tree = DecisionTreeRegressor(max_depth=l,random_state=0)
#     scores.append(CrossValid(tree, X_train, y_train))
# plt.plot(L, scores)
# plt.xticks(L)
# plt.ylabel(r"$R^2$ (10-fold cross validation)")
# plt.xlabel("max depth")
# plt.grid()
# plt.show()

# fit the model
tree = DecisionTreeRegressor(max_depth=3,random_state=0)
tree.fit(X_train, y_train)
R2_models["Tree"] = tree.score(X_test, y_test)
err = tree.predict(X_test)-y_test
err_models["Tree"] = (np.mean(err), np.var(err))

# ablation experiment
# tree.fit(X_reg_train, y_reg_train)
# tree_predict = inverse_boxcox(tree.predict(X_reg_test))
# print("tree score = {}".format(r2_score(y_test, tree_predict)))

'''Bagging lm'''
# selecting hyperparameter
# lm = LinearRegression()
# scores = []
# L = range(10, 101, 10)
# for l in L:
#     bagging_lm = BaggingRegressor(base_estimator=lm, n_estimators=l, random_state=0)
#     scores.append(CrossValid(bagging_lm, X_reg_train, y_reg_train))
# plt.plot(L, scores)
# plt.xticks(L)
# plt.ylabel(r"$R^2$ (10-fold cross validation)")
# plt.xlabel("number of estimators")
# plt.grid()
# plt.show()

# fit the model
bagging_lm = BaggingRegressor(base_estimator=lm, n_estimators=20, random_state=0)
bagging_lm.fit(X_reg_train, y_reg_train)
bagging_lm_predict = inverse_boxcox(bagging_lm.predict(X_reg_test))
R2_models["Bagging-LR"] = r2_score(y_test, bagging_lm_predict)
err = bagging_lm_predict-y_test
err_models["Bagging-LR"] = (np.mean(err), np.var(err))

# ablation experiment
# bagging_lm.fit(X_train, y_train)
# print("Bagging-LR score = {}".format(bagging_lm.score(X_test, y_test)))

'''Bagging mlp'''
# selecting hyperparameter
# mlp = MLPRegressor(hidden_layer_sizes=(4),  activation='relu', solver='adam', max_iter=5000, shuffle=True,random_state=0)
# scores = []
# L = range(10, 51, 5)
# for l in L:
#     bagging_mlp = BaggingRegressor(base_estimator=mlp, n_estimators=l, random_state=0)
#     scores.append(CrossValid(bagging_mlp, X_reg_train, y_reg_train))
# plt.plot(L, scores)
# plt.xticks(L)
# plt.ylabel(r"$R^2$ (10-fold cross validation)")
# plt.xlabel("number of estimators")
# plt.grid()
# plt.show()

# fit the model
bagging_mlp = BaggingRegressor(base_estimator=mlp, n_estimators=20, random_state=0)
bagging_mlp.fit(X_reg_train, y_reg_train)
bagging_mlp_predict = inverse_boxcox(bagging_mlp.predict(X_reg_test))
R2_models["Bagging-MLP"] = r2_score(y_test, bagging_mlp_predict)
err = bagging_mlp_predict-y_test
err_models["Bagging-MLP"] = (np.mean(err), np.var(err))

# ablation experiment
# bagging_mlp.fit(X_train, y_train)
# print("Bagging-MLP score = {}".format(bagging_mlp.score(X_test, y_test)))

'''Random Forest Regressor'''
# selecting hyperparameter
# scores = []
# L = range(10, 101, 10)
# for l in L:
#     rf = RandomForestRegressor(n_estimators=l,random_state=0)
#     scores.append(CrossValid(rf, X_train, y_train))
# plt.plot(L, scores)
# plt.xticks(L)
# plt.ylabel(r"$R^2$ (10-fold cross validation)")
# plt.xlabel("number of estimators")
# plt.grid()
# plt.show()

# fit the model
rf = RandomForestRegressor(n_estimators=40,random_state=0)
rf.fit(X_train, y_train)
R2_models["Random Forest"] = rf.score(X_test, y_test)
err = rf.predict(X_test)-y_test
err_models["Random Forest"] = (np.mean(err), np.var(err))

# ablation experiment
# rf.fit(X_reg_train, y_reg_train)
# rf_predict = inverse_boxcox(rf.predict(X_reg_test))
# print("Random Forest score = {}".format(r2_score(y_test, rf_predict)))

'''Stacking'''
# fit the model
mlp = MLPRegressor(hidden_layer_sizes=(4),  activation='relu', solver='adam', max_iter=10000, shuffle=True,random_state=0)
estimators = [('lm', lm), ('mlp', mlp), ('tree', tree)]
stacking = StackingRegressor(estimators=estimators)
stacking.fit(X_reg_train, y_reg_train)
stacking_predict = inverse_boxcox(stacking.predict(X_reg_test))
R2_models["Stacking"] = r2_score(y_test, stacking_predict)
err = stacking_predict-y_test
err_models["Stacking"] = (np.mean(err), np.var(err))

# ablation experiment
# stacking.fit(X_train, y_train)
# print("Stacking score = {}".format(stacking.score(X_test, y_test)))

'''GBoost'''
# selecting hyperparameter
# scores = []
# L = range(2, 21, 2)
# for l in L:
#     Gboost = GradientBoostingRegressor(n_estimators=l,random_state=0)
#     scores.append(CrossValid(Gboost, X_train, y_train))
# plt.plot(L, scores)
# plt.xticks(L)
# plt.ylabel(r"$R^2$ (10-fold cross validation)")
# plt.xlabel("number of estimators")
# plt.grid()
# plt.show()

# fit the model
Gboost = GradientBoostingRegressor(n_estimators=6, random_state=0)
Gboost.fit(X_train, y_train)
R2_models["GBoost"] = Gboost.score(X_test, y_test)
err = Gboost.predict(X_test)-y_test
err_models["GBoost"] = (np.mean(err), np.var(err))

# ablation experiment
# Gboost.fit(X_reg_train, y_reg_train)
# Gboost_predict = inverse_boxcox(Gboost.predict(X_reg_test))
# print("GBoost score = {}".format(r2_score(y_test, Gboost_predict)))

'''AdaBoost'''
# selecting hyperparameter
# scores = []
# L = range(2, 21, 2)
# for l in L:
#     Adaboost = AdaBoostRegressor(n_estimators=l,random_state=0)
#     scores.append(CrossValid(Adaboost, X_train, y_train))
# plt.plot(L, scores)
# plt.xticks(L)
# plt.ylabel(r"$R^2$ (10-fold cross validation)")
# plt.xlabel("number of estimators")
# plt.grid()
# plt.show()

# fit the model
Adaboost = AdaBoostRegressor(n_estimators=6, random_state=0)
Adaboost.fit(X_train, y_train)
R2_models["AdaBoost"] = Adaboost.score(X_test, y_test)
err = Adaboost.predict(X_test)-y_test
err_models["AdaBoost"] = (np.mean(err), np.var(err))

# ablation experiment
# Adaboost.fit(X_reg_train, y_reg_train)
# Adaboost_predict = inverse_boxcox(Adaboost.predict(X_reg_test))
# print("AdaBoost score = {}".format(r2_score(y_test, Adaboost_predict)))

'''plotting'''
print(err_models)
for key, value in err_models.items():
    plt.scatter(value[0], value[1], label=key)
plt.xlabel("error mean on test dataset")
plt.ylabel("error var on test dataset")
plt.grid()
plt.legend()
plt.show()

print(R2_models)
plt.bar(list(R2_models.keys()), list(R2_models.values()))
plt.xticks(rotation=30)
plt.ylabel(r"$R^2$ on test dateset")
plt.grid()
plt.show()

fig, ax = plt.subplots(3,3, figsize=(12,12),tight_layout=True,sharex=True,sharey=True)
ax[0,0].hist([y_test, lm_predict], label=["test dataset","estimated"], bins=20)
ax[0,0].set_title("Linear Rgression")
ax[0,1].hist([y_test, mlp_predict], label=["test dataset","estimated"], bins=20)
ax[0,1].set_title("MLP")
ax[0,2].hist([y_test, tree.predict(X_test)], label=["test dataset","estimated"], bins=20)
ax[0,2].set_title("Tree")
ax[1,0].hist([y_test, bagging_lm_predict], label=["test dataset","estimated"], bins=20)
ax[1,0].set_title("Bagging-LR")
ax[1,1].hist([y_test, bagging_mlp_predict], label=["test dataset","estimated"], bins=20)
ax[1,1].set_title("Bagging-MLP")
ax[1,2].hist([y_test, rf.predict(X_test)], label=["test dataset","estimated"], bins=20)
ax[1,2].set_title("Random Forest")
ax[2,0].hist([y_test, stacking_predict], label=["test dataset","estimated"], bins=20)
ax[2,0].set_title("Stacking")
ax[2,1].hist([y_test, Gboost.predict(X_test)], label=["test dataset","estimated"], bins=20)
ax[2,1].set_title("GBoost")
ax[2,2].hist([y_test, Adaboost.predict(X_test)], label=["test dataset","estimated"], bins=20)
ax[2,2].set_title("AdaBoost")
for i in range(3):
    for j in range(3):
        if j==0:
            ax[i,j].set_ylabel("Histogram")
        if i==2:
            ax[i,j].set_xlabel("y")
        ax[i,j].legend()
        ax[i,j].grid()
plt.show()

fig, ax = plt.subplots(3,3, figsize=(12,12),tight_layout=True,sharex=True,sharey=True)
ax[0,0].scatter(y_test, lm_predict)
ax[0,0].set_title("Linear Rgression")
ax[0,1].scatter(y_test, mlp_predict)
ax[0,1].set_title("MLP")
ax[0,2].scatter(y_test, tree.predict(X_test))
ax[0,2].set_title("Tree")
ax[1,0].scatter(y_test, bagging_lm_predict)
ax[1,0].set_title("Bagging-LR")
ax[1,1].scatter(y_test, bagging_mlp_predict)
ax[1,1].set_title("Bagging-MLP")
ax[1,2].scatter(y_test, rf.predict(X_test))
ax[1,2].set_title("Random Forest")
ax[2,0].scatter(y_test, stacking_predict)
ax[2,0].set_title("Stacking")
ax[2,1].scatter(y_test, Gboost.predict(X_test))
ax[2,1].set_title("GBoost")
ax[2,2].scatter(y_test, Adaboost.predict(X_test))
ax[2,2].set_title("AdaBoost")
for i in range(3):
    for j in range(3):
        if j==0:
            ax[i,j].set_ylabel("y (estimated)")
        if i==2:
            ax[i,j].set_xlabel("y (test dataset)")
        ax[i,j].grid()
plt.show()

# plt.hist([y_train, y_test], label=["train dataset", "test dataset"], density=True)
# plt.grid()
# plt.ylabel("Histogram")
# plt.xlabel("y")
# plt.legend()
# plt.show()