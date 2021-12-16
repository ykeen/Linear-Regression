import numpy as np
import pandas as pd


# Read Data From Excel
data = pd.read_csv(r'house_data.csv')
df = pd.DataFrame(data, columns=['grade', 'bathrooms', 'lat','sqft_living', 'view', 'price'])

# normalization
maxValue =df.max()
minValue = df.min()
meanValue = df.mean()
StdValue = df.std()
df = (df - df.min()) / (df.max() - df.min())
# print(df)


# Insert column It's value = one
df.insert(0, "ones", 1)
cols = df.shape[1]
rows =  df.shape[0]
X = df.iloc[:, 0: cols - 1]  # 0,1,2,3,4 predictor
y = df.iloc[:, cols - 1:cols]  # 5 target



# convert from data frames to matrices
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0, 0,0,0,0,0]))




# Cost Function
def costFunction(X, y, theta):
    J = np.power(((X * theta.T) - y), 2)
    return np.sum(J) / (2 * len(X))
print('Cost Function =', costFunction(X, y, theta))



# Gradeint descent fuction
def gradientDescent(X, y, theta, alpha, iterations):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iterations)

    for i in range(iterations):
        error = (X * theta.T) - y
        MSE = 1 / rows * sum(error)
        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

            theta = temp
            cost[i] = costFunction(X, y, theta)
        # print("erorr = ",error )
    return theta, cost , MSE


# call GD
alpha = 0.01
iterations = 100
returnedTheta, cost , error = gradientDescent(X, y, theta, alpha, iterations)
print("error = " , error)
c = costFunction(X, y, returnedTheta)
print("Cost  = ", c)
print("returend theta : ", returnedTheta)


# make predictions on new data
newdata = [11,4.5,47.6561,5420,0,1225000]
inputNormalize = (newdata - minValue) / (maxValue - minValue)
print("inputNormalize =" , inputNormalize)




for i in range(len(newdata)-1):
        priceOuput = sum(returnedTheta[0,i+1] * inputNormalize)


print("priceOuput = " , priceOuput+returnedTheta[0,0])


priceOuputDenormalize = priceOuput * (maxValue[len(maxValue)-1] - minValue[len(minValue)-1]) + minValue[len(minValue)-1]

print("predict price = " , priceOuputDenormalize)



