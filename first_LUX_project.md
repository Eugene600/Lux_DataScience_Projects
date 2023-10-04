# **Week One Project: Churn Prediction for Sprint**
 Imagine you're working with Sprint, one of the biggest telecom companies in the USA. They're really keen on figuring out how many customers might decide to leave them in the coming months. Luckily, they've got a bunch of past data about when customers have left before, as well as info about who these customers are, what they've bought, and other things like that. So, if you were in charge of predicting customer churn, how would you go about using machine learning to make a good guess about which customers might leave? What steps would you take to create a machine learning model that can predict if someone's going to leave or not?

 <u>What is a Churn Event</u>
 
  A Churn event refers to the loss of customers or subscribers for any reason at all. Businesses measure and track churn as a percentage of lost customers compared to total number of customers over a given time period.

 Predicting the churn event involved going through the following processes:

 <u>1. Data Collection</u>

First of all, I imported the pandas library as shown below.
```python
import pandas as pd
```

I loaded the dataset called Customer Churn which I extracted from Kaggle as shown below.
```python
data = pd.read_csv('Customer-Churn.csv')
```

Additionally, I did the following to show me the first five rows in the dataset.
```python
data.head()
```

Also, I extracted the churn column by doing the following:
```python
x = data.drop('Churn', axis=1)
y = data['Churn']
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
x = pd.get_dummies(x, drop_first=True)
```

<u>4. Splitting the data into training and testing sets</u>

I used the train_test_split function which is of the sklearn. model_selection package in Python splits arrays or matrices into random subsets for train and test data, respectively as shown below.
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
```

<u>5. Data Preprocessing</u>

I normalized the numeric features as shown below.
```python
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
```

<u>6. Creation and training of the logistic regression model</u>

```python
model = LogisticRegression()
model.fit(x_train, y_train)  
```

<u>7. Make predictions on the test set</u>

I made predictions of the train set and also to print the accuracy score. 
```python
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}') 
```

The output showed an accuracy score of 0.79 as shown below.

**Accuracy: 0.79**

I wrote the following code to make a classification report.
```python
print('\nClassification Report:')
print(classification_report(y_test, y_pred)) 
```

The following was the output.
![Alt text](<Classification Report for first_Lux_project.png>)






