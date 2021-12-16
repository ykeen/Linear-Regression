# Linear-Regression
Simple Linear regression and Multiple Linear regression with gradient descent to predict the price of houses

The attached dataset “house_data.csv” contains 21613 records of house sale prices. It includes homes sold between May 2014 and May 2015.

1- Simple Linear regression with gradient descent to predict the price based on sqft_living (Square footage of the apartments interior living space).
Given the hypothesis function: Y = C1 + C2 X
Y (target variable) = Price, X (predictor) = sqft_living, C1 and C2 are the parameters of the function.

2- Multiple Linear regression with gradient descent to predict price based on 5 predictors (grade, bathrooms, lat, sqft_living, view).
Given the hypothesis function: Y = C1 + C2 X2 + C3 X3 + C4 X4 + C5 X5 + C6 X6
Y (target variable) = Price, X (predictor) = (grade, bathrooms, lat, sqft_living, view), C1, C2, C3, C4, C5 and C6 are the parameters of the function.

a) Implement the gradient descent function to optimize parameters of the function.
b) Calculate error function to see how the error of the hypothesis function changes with every iteration of gradient descent.
c) Use optimized hypothesis function to make predictions on new data.
d) Try different values of learning rate and see how this changes the accuracy of the model.
