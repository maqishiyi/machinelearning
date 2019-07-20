import numpy as np
import matplotlib as plt

from sklearn.covariance import EmpiricalCovariance, MinCovDet

n_samples = 125 #125个训练集
n_outliers = 25 #25个异常点
n_features = 2  #二维的，方便平面展示

# generate data
gen_cov = np.eye(n_features)
gen_cov[0, 0] = 2
print(gen_cov)

# 随机数矩阵与上面的2*2矩阵点乘，就是为了把分布搞得像个椭圆
X = np.dot(np.random.randn(n_samples, n_features), gen_cov)
print(X)

# add some outliers
outliers_cov = np.eye(n_features)
outliers_cov[np.arange(1, n_features), np.arange(1, n_features)] = 7
print(outliers_cov)
X[-n_outliers:] = np.dot(np.random.randn(n_outliers, n_features), outliers_cov)
print(X)

#fit a minimum covariance determinant(MCD) robust estimator to data
robust_cov = MinCovDet().fit(X) #最小协方差确定

#compare estimators learned from the full data set with true parameters
emp_cov = EmpiricalCovariance().fit(X) # 最大协方差

