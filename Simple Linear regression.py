import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r'house_data.csv')
df = pd.DataFrame(data, columns=['sqft_living', 'price'])

# normalization
maxPrice = df.price.max()
minPrice = df.price.min()
maxSqft = df.sqft_living.max()
minSqft = df.sqft_living.min()
sqftMean = df.sqft_living.mean()
sqftStd = df.sqft_living.std()
priceMean = df.price.mean()
priceStd = df.price.std()
df['sqft_living'] = (df['sqft_living'] - df.sqft_living.min()) / (df.sqft_living.max() - df.sqft_living.min())
df['price'] = (df['price'] - df.price.min()) / (df.price.max() - df.price.min())


# print(df)



df.insert(0, "ones", 1)
cols = df.shape[1]
rows =  df.shape[0]
X = df.iloc[:, 0: cols - 1]  # 0,1
y = df.iloc[:, cols - 1:cols]  # yet colomn 2 which is y


print("X \n", X)
# print("Y \n", y)

print("*******************")
# convert from data frames to matrices
X = np.matrix(X.values)  # divided by 100000 ?!!
y = np.matrix(y.values)
theta = np.matrix(np.array([0, 1]))
print("theta = ", theta)



print("X \n", X)
print("Y \n", y)


# Cost Function
def costFunction(X, y, theta):
    J = np.power(((X * theta.T) - y), 2)

    return np.sum(J) / (2 * len(X))


print('Cost Function =', costFunction(X, y, theta))

""""""


# Gradeint descent fuction
def gradientDescent(X, y, theta, alpha, iterations):
    temp = np.matrix(np.zeros(theta.shape))
    print("temp theta = ", temp)
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iterations)

    for i in range(iterations):
        error = (X * theta.T) - y
        MSE = 1 / rows * sum(error)
        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            # print("x length = ", len(X))

            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

            theta = temp
            cost[i] = costFunction(X, y, theta)

    return theta, cost, MSE


# Initializing num for learning rate(alpha) and iterations
alpha = 0.003
iterations = 100

# call GD
returnedTheta, cost , error = gradientDescent(X, y, theta, alpha, iterations)
print("error =" ,error)
c = costFunction(X, y, returnedTheta)
# print("Cost  = ", format(c, 'f'))
print("Cost  = ", c)

print("returend theta : ", returnedTheta)


# try to plot best fit line
f = returnedTheta[0, 0] + returnedTheta[0, 1] * X
inputNormalize = (1180 - minSqft) / (maxSqft - minSqft)


priceOuput = returnedTheta[0, 0] + returnedTheta[0, 1] * inputNormalize

priceOuputDenormalize = priceOuput * (maxPrice - minPrice) + minPrice



print("predict price = " , priceOuputDenormalize)
"""
fig, ax = plt.subplots(figsize = (5, 5))
ax.plot(X,f,'r',label = 'predection')
ax.scatter(df.sqft_living, df.price, label = 'training data')
ax.legend(loc = 2)
ax.set_xlabel('sqft_living')
ax.set_ylabel('price')
plt.show()

"""
