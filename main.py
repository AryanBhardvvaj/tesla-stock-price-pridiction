# for numarcal computation,manupulation and visualization
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# install plotly for building graphs
 
import plotly as py
import plotly.graph_objs as go
from plotly.offline import plot

#for offline plotting
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

tesla = pd.read_csv("tesla.csv")
print(tesla.head(5))
# print(tesla.info()) #shows info about rows cols memory etc
tesla['Date']=pd.to_datetime(tesla['Date']) #converting date to datetime 


# calculating total no of days

print(f'Dataframe contains stock prices between {tesla.Date.min()} {tesla.Date.max()}') 
print(f'Total days = {(tesla.Date.max()  - tesla.Date.min()).days} days')

# describe function gives you the count mean quatile max min std dev

print(tesla.describe())

#creating a box plot of each 'Open','High','Low','Close','Adj Close'

boxpot1=tesla[['Open','High','Low','Close','Adj Close']]
plt.boxplot(boxpot1)
plt.show()


# now we will plot a graph using plotly libraray

# Setting the layout for our plot using the go layout function
#and providing a title to the graph 
# x axis title to date x axis 
# also the title font is defined famiy size and color
# similarly y axis is specified the same the title is set to price


layout = go.Layout(
    title='Stock Prices of Tesla',
    xaxis=dict(
        title='Date',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='Price',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)



tesla_data = [{'x':tesla['Date'], 'y':tesla['Open']}]

# passing the data list and the layout created to plot variable

plot1 = go.Figure(data=tesla_data, layout=layout)

#now plotting the graph using i plot function

iplot(plot1)

# now to build the lenear reb\gressin model -> >

# Building the regression model
from sklearn.model_selection import train_test_split
#train test split is a function

#For preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
#these above are preprocessing functions 

#For model evaluation and finding the accuracy we will use mean sq error and r squared error
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

#spliting the data into train and test sets
'''x has the independent variables and y has the dependent variables
i e the close call % of the total and assigning  a random state of 101''' 

X = np.array(tesla.index).reshape(-1,1)
Y = tesla['Close']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=200)

# now wwe will perfom feature scaling on the platfom
#for that we use standeredscaler.fit function
#and pass our x train data

scaler = StandardScaler().fit(X_train)

#now import the lenear regression function

from sklearn.linear_model import LinearRegression

#using a var lm we declare a leniar regression function
# and then use fit method to pass the training data

lm = LinearRegression()
lm.fit(X_train, Y_train)

#now we have created a lenier regression mfunction

#now we will plot the graph between actual and pridicted data set
#we will create a scatter plot and draw a scatter line
# we haveused tohe go.scatter function 
# we have used markers to pridict actual and line for pridicted

trace0 = go.Scatter(
    x = X_train.T[0],
    y = Y_train,
    mode = 'markers',
    name = 'Actual'
)
trace1 = go.Scatter(
    x = X_train.T[0],
    y = lm.predict(X_train).T,
    mode = 'lines',
    name = 'Predicted'
)
tesla_data = [trace0,trace1]
layout.xaxis.title.text = 'Day'
plot2 = go.Figure(data=tesla_data, layout=layout)

iplot(plot2)


#Calculate scores for model evaluation
#we are using 2 matrix one is r sq errror and mean sq error 
#for r sq  we use r2_score function and pass y and x train using lmtrain function
# similatrly for test data we do the same
#similarly we calculate mse 

scores = f'''
{'Metric'.ljust(10)}{'Train'.center(20)}{'Test'.center(20)}
{'r2_score'.ljust(10)}{r2_score(Y_train, lm.predict(X_train))}\t{r2_score(Y_test, lm.predict(X_test))}
{'MSE'.ljust(10)}{mse(Y_train, lm.predict(X_train))}\t{mse(Y_test, lm.predict(X_test))}
'''
print(scores)
