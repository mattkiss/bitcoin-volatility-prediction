import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import json

# Parameters

numOfRendements=10      # Number of returns used to compute a proxy (there are 4221 prices in the batch)
numOfColumnsX=3         # Number of columns of a sample (with the intercept)
timeBetweenPrices=3     # Interval between two prices
mu=1                    # Forgetting factor of the RLS algorithm

# Computation of the proxies

messages = []
with open('messages.txt') as f:
    for line in f:
        messages.append(json.loads(line))

proxies=[]
firstMessage=messages.pop(0)
lastPrice=float(firstMessage['price'])
lastTime=int(firstMessage['timestamp'])
rendements=np.empty(0)
while messages:
    newMessage=messages.pop(0)
    prices=[lastPrice, float(newMessage['price'])]
    times=[lastTime, int(newMessage['timestamp'])]
    interpolatedTimes=list(range(lastTime, int(newMessage['timestamp']), timeBetweenPrices))
    interpolatedPrices=list(np.interp(interpolatedTimes, times, prices))
    while interpolatedTimes:
      newPrice=interpolatedPrices.pop(0)
      rendement=np.log(newPrice/lastPrice)
      lastPrice=newPrice
      lastTime=interpolatedTimes.pop(0)
      rendements=np.append(rendements, rendement)
      if rendements.size==numOfRendements:
        proxies.append(np.sum(np.square(rendements))/numOfRendements)
        rendements=rendements[1:]

# Creation of the np.arrays

X=()
Y=()
for i in range(len(proxies)-numOfColumnsX+1): # +1 because numOfColumnsX takes the intercept into account
     X+=(proxies[i:i+numOfColumnsX-1],)       # -1 because numOfColumnsX takes the intercept into account
     Y+=([proxies[i+numOfColumnsX-1]],)

X_train=X[:len(X)//2]
X_test=X[len(X)//2:]
Y_train=Y[:len(Y)//2]
Y_test=Y[len(Y)//2:]

X_train=np.array(X_train)
X_test=np.array(X_test)     
Y_train=np.array(Y_train)
Y_test=np.array(Y_test) 

# Normalisation des np.arrays

X_train=(X_train-X_train.mean(0))/X_train.std(0)
X_test=(X_test-X_test.mean(0))/X_test.std(0)    
Y_train=(Y_train-Y_train.mean(0))/Y_train.std(0)
Y_test=(Y_test-Y_test.mean(0))/Y_test.std(0)

X_train=np.c_[np.ones(len(X_train)), X_train]  # Adding of the intercept
X_test=np.c_[np.ones(len(X_test)), X_test]

# Linear regression

regr = linear_model.LinearRegression(fit_intercept=False)
regr.fit(X_train, Y_train)

# RLS algorithm

beta=np.zeros((numOfColumnsX,1))
V=np.diag(np.zeros(numOfColumnsX)+1)    
for i in np.arange(X_train.shape[0]):
    x=X_train[i,:]
    x.shape=(1,numOfColumnsX);
    err=Y_train[i]-x.dot(beta)
    V=1.0/mu*(V-V.dot(x.T).dot(x).dot(V)/(1.0+float(x.dot(V).dot(x.T)))) # dot = matrix multiplication
                                                                         # .T = Transpose
    gamma=V.dot(x.T)
    beta=beta+gamma*err

# Naive forecaster: last value

Y_pred_last=[]
for i in np.arange(X_test.shape[0]):
    x=X_test[i,:]
    Y_pred_last.append([x[-1]])
Y_pred_last=np.array(Y_pred_last)

# Naive forecaster: mean

Y_pred_mean=[]
for i in np.arange(X_test.shape[0]):
    x=X_test[i,:]
    Y_pred_mean.append([np.mean(x[1:])])
Y_pred_mean=np.array(Y_pred_mean)

# Display parameters

print("Parameters\n----------")
print(numOfRendements, "returns are used to compute a proxy")
print(numOfColumnsX, "is the size of a sample (with the intercept)")
print(timeBetweenPrices, "seconds is the interval used to linearly interpolate prices")
print(len(proxies), "proxies have been computed and used")
print(len(X_train), "is the size of the training and testing test\n")

# Display coefficients

print("Coefficients\n------------")
print("Linear regression", regr.coef_)
print("RLS              ", beta.T, "\n")

# Linear model prediction

print("Linear regression quality\n-------------------------")
Y_pred=regr.predict(X_test)
print("Mean squared error: %.10f" % mean_squared_error(Y_test, Y_pred))
print("Normalized mean squared error (last): %.10f" % np.divide(mean_squared_error(Y_test, Y_pred),mean_squared_error(Y_test, Y_pred_last)))
print("Normalized mean squared error (mean): %.10f" % np.divide(mean_squared_error(Y_test, Y_pred),mean_squared_error(Y_test, Y_pred_mean)))
print("Coefficient of determination: %.10f" % r2_score(Y_test, Y_pred), "\n")

# RLS algorithm prediction

print("RLS quality\n-----------")
Y_pred_RLS=X_test.dot(beta)
print("Mean squared error: %.10f" % mean_squared_error(Y_test, Y_pred_RLS))
print("Normalized mean squared error (last): %.10f" % np.divide(mean_squared_error(Y_test, Y_pred_RLS),mean_squared_error(Y_test, Y_pred_last)))
print("Normalized mean squared error (mean): %.10f" % np.divide(mean_squared_error(Y_test, Y_pred_RLS),mean_squared_error(Y_test, Y_pred_mean)))
print("Coefficient of determination: %.10f" % r2_score(Y_test, Y_pred_RLS))