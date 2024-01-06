# Introduction

本项目的目的是设计集成学习模型来预测治疗后的指标评分，其中，自变量由 31 个变量构成，包括 30 个治疗前指标评分(数值型)和所用的治疗药物(分类型)，因变量是治疗后的指标总分(数值型). 数据集分为训练集(550 条记录)和测试集(137 条记录)，模型要在训练集上训练，并在测试集上测试模型的预测效果.

本代码主要包含如下操作：

- 数据预处理

  - 缺失数据处理
  - 数据可视化
  - 离散型变量数值化
  - 特征选择
    - 数值型变量的多重共线性诊断
    - 药物治疗效果的 Wilcoxon 秩和检验
    - 基于 Lasso 回归和 Box-Cox 变换的变量筛选

- 模型设计与超参数选择

  - Linear Regression
  - MLP
  - Tree
  - Bagging-LR
  - Bagging-MLP
  - Random Forest
  - Stacking
  - GBoost
  - AdaBoost

- 绘制结果图

# Configuration instructions

代码文件采用 python 3.8.15 语言, 运行前请确认已安装如下的库：

```python
numpy
matplotlib
pandas
scipy
sklearn
```

# Operating instructions

代码中对前述操作均有详细的注释.

例如，在训练 MLP 时，选择超参数的代码如下：

```python
scores = []
L = range(5, 51, 5)
for l in L:
    mlp = MLPRegressor(hidden_layer_sizes=(l), activation='logistic', solver='adam', max_iter=10000, shuffle=True,random_state=0)
    scores.append(CrossValid(mlp, X_reg_train, y_reg_train))
#plotting
plt.plot(L, scores)
plt.xticks(L)
plt.ylabel(r"$R^2$ (10-fold cross validation)")
plt.xlabel("hidden layer sizes")
plt.grid()
plt.show()
```

基于原自变量集合的拟合操作如下:

```python
mlp = MLPRegressor(hidden_layer_sizes=(30), activation='logistic', solver='adam', max_iter=10000, shuffle=True,random_state=0)
mlp.fit(X_train, y_train)
print("mlp score = {}".format(mlp.score(X_test, y_test)))
```

基于筛选后的自变量集合和 Box-Cox 变换的拟合操作如下:

```python
mlp = MLPRegressor(hidden_layer_sizes=(30), activation='logistic', solver='adam', max_iter=10000, shuffle=True,random_state=0)
mlp.fit(X_reg_train, y_reg_train)
mlp_predict = inverse_boxcox(mlp.predict(X_reg_test))
print("mlp score = {}".format(r2_score(y_test, mlp_predict)))
```
