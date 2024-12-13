高斯混合模型假设一个多变量的数据，是来自于多个不同的正态分布

# 情况1，当知道几个成分，训练数据里带有标签

这种情况处理很简单，数据点x预测为A类的概率 = A的先验概率 * 数据x属于A的密度 ， $P(Class_a|x) = \pi_{a}*f(x|Class_a)$ , $\pi_{a}是先验概率$, 本来严格来说 把密度f定积分过后才是概率，但是为了方便，其实不积分直接比较是一样的

$密度f的公式：  f(x属于A组) = \frac{1}{\sqrt{(2\pi)^d |\Sigma_A|}} \exp\left(-\frac{1}{2}(x - \mu_A)^T \Sigma_A^{-1} (x - \mu_A)\right)$

$x$ 是要预测的那一行数据向量

$\mu$ 是当前类别均值向量

$d$ 是数据的维度，也就是有几个变量

$|\sigma|$ 是协方差矩阵的行列式，也就是determinant

## 例子

```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
train_data = pd.read_csv(f'D:/Download/Training set.csv')
# 把女性换为1，男性换位0
train_data['Sex'] = train_data['Sex'].replace({'Female':1,'Male':0})
# 分为男性数据和女性数据
male_data = train_data[train_data['Sex'] == 0]
female_data = train_data[train_data['Sex'] == 1]
# 计算男性的均值向量和协方差矩阵
male_mean = [male_data.iloc[:,0].mean(),male_data.iloc[:,1].mean()]
male_cov = np.cov(male_data.iloc[:, 0], male_data.iloc[:, 1])
# 去掉一个outlier
female_data = female_data[female_data['Height'] <= 250]
# 计算女性的均值向量和协方差矩阵
female_mean = [female_data.iloc[:,0].mean(),female_data.iloc[:,1].mean()]
female_cov = np.cov(female_data.iloc[:, 0], female_data.iloc[:, 1])
# 男女的先验概率，既是训练集出现比例
f_prior_p = len(female_data)/len(train_data)
m_prior_p = 1-f_prior_p
```

## 建立模型
```
class GaussianModel:
    # 要求初始参数
    def __init__(self, male_mean, female_mean, male_cov, female_cov, prior_prob_m, prior_prob_f):
        self.male_m = np.array(male_mean)
        self.female_m = np.array(female_mean)
        self.male_cov = np.array(male_cov)
        self.female_cov = np.array(female_cov)
        self.prior_prob_m = prior_prob_m
        self.prior_prob_f = prior_prob_f
    # 计算概率
    def predict(self, x):
        x = np.array(x) 
        result = []
        male_cov_det = np.linalg.det(self.male_cov)
        female_cov_det = np.linalg.det(self.female_cov)
        male_cov_inv = np.linalg.inv(self.male_cov)
        female_cov_inv = np.linalg.inv(self.female_cov)
        for row in x:
            diff_m = row - self.male_m
            m_density = (
                1 / np.sqrt((2 * np.pi) ** len(row) * male_cov_det)
                * np.exp(-0.5 * diff_m @ male_cov_inv @ diff_m)
            )
            diff_f = row - self.female_m
            f_density = (
                1 / np.sqrt((2 * np.pi) ** len(row) * female_cov_det)
                * np.exp(-0.5 * diff_f @ female_cov_inv @ diff_f)
            )
            pro_m = self.prior_prob_m * m_density
            pro_f = self.prior_prob_f * f_density
            # 哪一类概率大就分配为哪一类
            if pro_m < pro_f:
                result.append(1)  # Female
            else:
                result.append(0)  # Male
        return result
```

## 预测结果

```
test_data = pd.read_csv(f'D:/Download/Test set.csv')
test = test_data.iloc[:,:2]
labels = test_data.iloc[:,2]
labels = labels.replace({'Female':1,'Male':0})
labels = np.array(labels)
model = GaussianModel(male_mean,female_mean,male_cov,female_cov,m_prior_p,f_prior_p)
prediction = model.predict(test)
from sklearn.metrics import confusion_matrix
confusion_matrix(labels,prediction)
```

$$混淆矩阵： \begin{bmatrix}94 & 12 \\
15 & 84\end{bmatrix}$$

