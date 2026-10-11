import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

#1.LogisticRegression()
#load dataset
data=load_breast_cancer()
X=pd.DataFrame(data.data,columns=data.feature_names)
y=data.target
print(X.shape)

#split data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#train model
lr=LogisticRegression(max_iter=5000)
lr.fit(X_train,y_train)
print("model trained")

#first test point
x_e=X_test.iloc[0]

#prediction probability
pred=lr.predict_proba(x_e.to_frame().T)[0,1]  #取第0行第1列，也就是类别1（benigh良性）的概率
print("Prediction:",pred)    #第一个病人是良性的概率

#2.Random Forest
from sklearn.ensemble import RandomForestClassifier
#random forest model
rf=RandomForestClassifier(n_estimators=200,random_state=0)
rf.fit(X_train,y_train)
print("Random Forest trained")
pre_rf=rf.predict_proba(x_e.to_frame().T)[0,1]
print("RF Prediction:",pre_rf)  #随机森林预测良性的概率

#3.SHAP
import shap
import matplotlib.pyplot as plt
#create explainer
explainer=shap.TreeExplainer(rf)  #创建一个解释器
#calculate shap values
shap_values=explainer(X_test)  #对测试集计算shap值
print(type(shap_values))
print(len(shap_values))     #114,(569*0.2)
print(shap_values[1].shape)   #(30,2) 就是30行*2列   第二个病人的SHAP数据（30个eature，2个类别）
shap_values_class1=shap_values[:,:,1]    #[病人，feature，class(1就是良性）]
print(shap_values.shape)   #（114，30，2）   全部病人的SHAP数据
print(shap_values_class1.shape)  #（114，30）  只有一个类别（良性）
#bar plot
shap.plots.bar(shap_values_class1, show=False)   #global explanation 解释全部病人  #figure1
plt.subplots_adjust(left=0.35)
plt.savefig("Tree_global.png",dpi=300,bbox_inches='tight')
plt.show()
#explain first patient
shap.plots.waterfall(shap_values_class1[0], show=False)  #local explanation 解释一个病人  #figure2
plt.gcf().set_size_inches(12,6)
plt.subplots_adjust(left=0.35,right=0.90,bottom=0.25)
plt.savefig("Tree_local.png",dpi=300,bbox_inches='tight',pad_inches=0.5)
plt.show()
#4.kernelshap
import shap
# background dataset
background = X_train.sample(50, random_state=0)
# create KernelSHAP explainer
explainer_kernel = shap.KernelExplainer(
    lr.predict_proba,
    background)
print("KernelSHAP explainer created")
# calculate shap values
shap_values_kernel = explainer_kernel.shap_values(
    X_test.iloc[:10])
print("KernelSHAP finished")
print(type(shap_values_kernel))
print(len(shap_values_kernel))
print(np.array(shap_values_kernel).shape)

# class 1 (benign)
shap_values_kernel_class1 = shap_values_kernel[:, :, 1]
# # bar plot
# shap.summary_plot(
#     shap_values_kernel_class1,
#     X_test.iloc[:10],
#     plot_type="bar")  #figure3  global

#5.LinearSHAP
explainer_linear = shap.LinearExplainer(
    lr,
    X_train)
shap_values_linear = explainer_linear.shap_values(X_test)
print("LinearSHAP finished")
print(np.array(shap_values_linear).shape)
# shape
print(type(shap_values_linear))
print(np.array(shap_values_linear).shape)
# bar plot
shap.summary_plot(
    shap_values_linear,
    X_test,
    plot_type="bar")  #figure4


