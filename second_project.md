# **Week Two Project: Linear and Random Forest Regression
Let’s say we want to build a model to predict booking prices on Airbnb. Between linear regression and random forest regression, which model would perform better and why?

<u>What is Linear Regression?</u>

Linear regression analysis is used to predict the value of a variable based on the value of another variable. The variable you want to predict is called the dependent variable. The variable you are using to predict the other variable's value is called the independent variable.

<u>What is Random Forest Regression?</u>

Random forest regression is a supervised learning algorithm and bagging technique that uses an ensemble learning method for regression in machine learning. The trees in random forests run in parallel, meaning there is no interaction between these trees while building the trees.

 Predicting the booking prices involved going through the following processes:

 <u>1. Data Collection</u>

First of all, I imported the following libraries to  enable me to do both Linear and Random Forest Regression as shown below.
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
```

I loaded the dataset called data.csv which I extracted from Kaggle as shown below.
```python
data = pd.read_csv('data.csv')
```

Additionally, I did the following to show me the first five rows in the dataset.
```python
data.head()
```

Also, I extracted the churn column for both Linear and Random Forest Regression by doing the following:
```python
x = data.drop(['date', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
       'floors', 'waterfront', 'view', 'condition', 'sqft_above',
       'sqft_basement', 'yr_built', 'yr_renovated', 'street', 'city',
       'statezip', 'country'], axis=1)
y = data['price']
```

<u>2. Handling missing values</u>

I used dropna() which  removes rows that contain null values.
I did this as shown below.
```python
data.dropna(inplace=True)
```

<u>3. Hot Encoding</u>

Hot Encoding is a technique used to encode categorical data to numerical ones either 0 or 1.

I used the following code to perform Hot encoding.
```python
X = pd.get_dummies(X, drop_first=True)
```

<u>4. Splitting the data into training and testing sets</u>

I used the train_test_split function which is of the sklearn. model_selection package in Python splits arrays or matrices into random subsets for train and test data, respectively as shown below.
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

<u>5. Data Preprocessing</u>

I normalized the numeric features as shown below.
```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

<u>6. Creation and training of the logistic regression model</u>

```python
# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Create and train the Random Forest Regression model
model = RandomForestRegressor(n_estimators=100, random_state=42) 
model.fit(X_train, y_train)
```

<u>7. Make predictions on the test set</u>

I made predictions of the train set . 
```python
y_pred = model.predict(X_test) 
```

<u>8. Model Evaluation</u>

```python
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')
```

<u>9. Data Visualization</u>

I used scatter plot 
```python
import matplotlib.pyplot as plt

# Visualize the predictions
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual Prices vs. Predicted Prices')
plt.show()
```

The scatter plot for the Linear Forest regression is as follows:
![Alt text](image.png)

The scatter plot for the Random Forest regression is as follows:
![Alt text](image-1.png)


<u>10. What is Mean Squared Error</u>

The Mean Squared Error measures how close a regression line is to a set of data points. Mean square error is calculated by taking the average, specifically the mean, of errors squared from data as it relates to a function.

<u>11. What is R squared</u>

R-Squared (R² or the coefficient of determination) is a statistical measure in a regression model that determines the proportion of variance in the dependent variable that can be explained by the independent variable.

In other words, r-squared shows how well the data fit the regression model (the goodness of fit).


From the Linear Regression Model, I got the following results:

```
Mean Squared Error: 0.00
R-squared: 1.00
```

From the Random Forest Regression Model, I got the following results:

```
Mean Squared Error: 517916348987.77
R-squared: 0.49
```


<u>Which is the better model?</u>

From the above data, Linear Regression is better than Random Forest Regression since the Mean Squared error and R-squared is better than that of Random Forest regression. This shows that our data has a linear relationship hence Linear Regression is the better model in this case.






