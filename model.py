

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


data=pd.read_csv('Weight_Index.csv')
data['Gender1']=data['Gender'].map({'Male':1,'Female':0})
data['diff']=data['Height']-data['Weight']
data['suma']=data['Height']+data['Weight']
data['mult']=data['Height']*data['Weight']
data['imc']=data['Weight']/(data['Height']/100)**2
data['div']=data['Weight']/data['Height']

x,y=data.drop(['Index','Gender'],axis=1),data['Index']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.35,random_state=2020810)

from sklearn.linear_model import LogisticRegression
mod4=LogisticRegression(multi_class='multinomial',solver ='newton-cg').fit(x_train,y_train)
pred4=mod4.predict(x_test)
confusion_matrix(y_test,pred4)
import pickle
pickle.dump(mod4, open('model.pkl','wb'))
np.argmin(pred4)
x_test.iloc[22,0:]
mod4.predict([x_test.iloc[22,0:]])
