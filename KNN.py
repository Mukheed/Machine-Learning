#KNN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("/content/User_Data.csv")
x=df.iloc[:,[2,3]].values
y=df.iloc[:,4].values
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
st=StandardScaler()
x_train=st.fit_transform(x_train)
x_test=st.transform(x_test)
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)
plt.scatter(x[:, 0], x[:,1], c=y, cmap='viridis', s=30)
plt.xlabel('Age')
plt.ylabel('Salary')
plt.title('KNN Results')
plt.show()