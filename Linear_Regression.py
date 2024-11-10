#Linear Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("/content/Salary_Data.csv")
x=df[['YearsExperience']]
y=df[['Salary']]
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=0)
r=LinearRegression()
r.fit(x_train,y_train)
y_p=r.predict(x_test)
x_p=r.predict(x_train)
plt.scatter(x_train,y_train)
plt.plot(x_train,x_p,color='red')
plt.title('Linear Regression')
plt.show()
plt.scatter(x_test,y_test)
plt.title("Linear regression")
plt.plot(x_train,x_p,color='red')
