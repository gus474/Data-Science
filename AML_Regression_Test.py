import statsmodels
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,precision_score,auc,roc_auc_score,roc_curve,recall_score
#Sklearning packages
#read the data
dataset=pd.read_csv('C:\\Users\\User\Downloads\\datasets_606091_1086781_ML2.csv')
dataset.head()

#Clean the data
dataset.drop('nameOrig',axis=1,inplace=True)
dataset.drop('isFlaggedFraud',axis=1,inplace=True)
dataset.drop('nameDest',axis=1,inplace=True)
dataset.head()
sample_dataframe=dataset.sample(n=100000)
X=sample_dataframe.iloc[:,:-1].values
y=sample_dataframe.iloc[:,7].values
labelencoder=LabelEncoder()
X[:,1]=labelencoder.fit_transform(X[:,1])

#Split training and testing data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)
X_test,X_val,y_test,y_val=train_test_split(X_test,y_test,test_size=0.5,random_state=1)
counts=np.unique(y_train,return_counts=True)

#Run the Regression Model
model=linear_model.LogisticRegression(random_state=0,solver='liblinear',multi_class='auto').fit(X,y)
print('intercept:',model.intercept_)
print('slope:',model.coef_)
y_pred=model.predict(X_test)
print('predicted response:',y_pred,sep='\n')
r_sq=model.score(X_train,y_train)
print('coefficient of determination:',r_sq)
df=pd.DataFrame({'Actual':y_test.flatten(),'Predicted':y_pred.flatten()})
df

#Plotting the data
df1=df.head(100)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major',linestyle='-',linewidth='0.5',color='green')
plt.grid(which='minor',linestyle=':',linewidth='0.5',color='black')
plt.show()
cm=confusion_matrix(y_test,y_pred)
roc=roc_auc_score(y_test,y_pred)
fpr,tpr,thresholds=roc_curve(y_test,y_pred)
roc_auc=auc(fpr,tpr)#Plot ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr,tpr,'b',label='AUC = %0.2f'%roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
