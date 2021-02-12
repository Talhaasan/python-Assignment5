import pandas as pd
import matplotlib.pyplot as pl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn import metrics

fileURL = 'C:\\Users\\pc\\.spyder-py3\\openCV\\iris.data'

iris = pd.read_csv (fileURL , names=[ 'Sepal Length' , 'Sepal Width' ,
                                      'Petal Length' ,  'Petal Width' , 
                                      'Species' ] , header=None ) 
iris = iris.dropna() 

def pairs(data):  
    i = 1
    # Divide columns into features and class
    features = list(data.columns)
    classes = features[-1] # create class column
    del features[-1] # delete class column from feature vector
    # Generate an nxn subplot figure, where n is the number of features
    figure = pl.figure(figsize=(5*(len(data.columns)-1), 4*(len(data.columns)-1)))
    for col1 in data[features]:
        for col2 in data[features]:
            ax = pl.subplot(len(data.columns)-1, len(data.columns)-1, i)
            if col1 == col2:
                ax.text(2.5, 4.5, col1, style='normal', fontsize=20)
                ax.axis([0, 10, 0, 10])
                pl.xticks([]), pl.yticks([])
            else:
                for name in data[classes]:
                    cond = data[classes] == name
                    ax.plot(data[col2][cond], data[col1][cond], linestyle='none', marker='o', label=name)
                #t = plt.title(name)
            i += 1
    pl.show()

#pairs(iris)

def showingCorrelation(iris):

    pl.xlabel('Features')
    pl.ylabel('Species')
    
    plX = iris.loc[:,'Sepal Length']
    plY = iris.loc[:,'Species']
    pl.scatter(plX,plY,color='blue',label = 'Sepal Length')
    
    plX = iris.loc[:,'Sepal Width']
    plY = iris.loc[:,'Species']
    pl.scatter(plX,plY,color='green',label = 'Sepal Width')
    
    plX = iris.loc[:,'Petal Length']
    plY = iris.loc[:,'Species']
    pl.scatter(plX,plY,color='red',label = 'Petal Length')
    
    plX = iris.loc[:,'Petal Width']
    plY = iris.loc[:,'Species']
    pl.scatter(plX,plY,color='black',label='Petal Width')
    
    pl.legend(loc=4,prop={'size':8})
    pl.show()
      
#showingCorrelation(iris)

def applyLinearRegressionWithSepalLengthAndPetalLength(iris):
    sepal_length=iris.loc[:,'Sepal Length']
    pedal_length=iris.loc[:,'Petal Length']
    
    label_Encoder=preprocessing.LabelEncoder()
    
    iris_X = np.column_stack((sepal_length,pedal_length))
    iris_y = label_Encoder.fit_transform(iris.iloc[:,-1])

    iris_X_train,iris_X_test,iris_y_train,iris_y_test=train_test_split(iris_X,iris_y,test_size=0.2,random_state=0)

    regr = LinearRegression()

    regr.fit(iris_X_train,iris_y_train)

    y_pred = regr.predict(iris_X_test)
    
    df = pd.DataFrame({'Actual': iris_y_test.flatten(), 'Predicted': y_pred.flatten()})
    
    print ("Coefficients : \n" , regr.coef_)
    print ( "Residual sum of squares : %.2f" % 
    np .mean ((regr.predict ( iris_X_test ) - iris_y_test)** 2))      
    print ( "Variance score : %.2f" % regr.score ( iris_X_test , iris_y_test)) 
    print()
    print('Mean Absolute Error:', metrics.mean_absolute_error(iris_y_test, y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(iris_y_test, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(iris_y_test, y_pred)))
    print()
    print(df)
    
    df1 = df.head(25)
    df1.plot(kind='bar',figsize=(16,10))
    pl.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    pl.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    pl.title('Linear Regression With Sepal Length and Petal Length')
    pl.show()
        
#applyLinearRegressionWithSepalLengthAndPetalLength(iris)
    
def applyLinearRegressionWithSepalWidthAndPetalLength(iris):
    sepal_width=iris.loc[:,'Sepal Width']
    pedal_length=iris.loc[:,'Petal Length']
    
    label_Encoder=preprocessing.LabelEncoder()
    
    iris_X = np.column_stack((sepal_width,pedal_length))
    iris_y = label_Encoder.fit_transform(iris.iloc[:,-1])
  

    iris_X_train,iris_X_test,iris_y_train,iris_y_test=train_test_split(iris_X,iris_y,test_size=0.2,random_state=0)

    regr = LinearRegression()

    regr.fit(iris_X_train,iris_y_train)

    y_pred = regr.predict(iris_X_test)
    
    df = pd.DataFrame({'Actual': iris_y_test.flatten(), 'Predicted': y_pred.flatten()})

    print ("Coefficients : \n" , regr.coef_)
    print ( "Residual sum of squares : %.2f" % 
    np .mean ((regr.predict ( iris_X_test ) - iris_y_test)** 2))      
    print ( "Variance score : %.2f" % regr.score ( iris_X_test , iris_y_test))
    print()
    print('Mean Absolute Error:', metrics.mean_absolute_error(iris_y_test, y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(iris_y_test, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(iris_y_test, y_pred)))
    print()
    print(df)
    
    df1 = df.head(25)
    df1.plot(kind='bar',figsize=(16,10))
    pl.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    pl.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    pl.title('Linear Regression With Sepal Width and Petal Length')
    pl.show()
    
applyLinearRegressionWithSepalWidthAndPetalLength(iris)





