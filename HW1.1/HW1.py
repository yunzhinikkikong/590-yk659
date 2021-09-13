#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 14:33:36 2021

@author: Nikkikong
"""
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from scipy.optimize import minimize

###################
###Read Data
###################


### Make a class called “Data” which reads the json

class Data:
       
    def __init__(self,file):
        with open(file) as f:
            data=f.read()
        allvars = json.loads(data)    
        self.is_adult = allvars["is_adult"]
        self.weight = allvars["y"]
        self.age = allvars["x"]

dataset=Data('weight.json') 
# Convert to a pandas dataframe
df = pd.DataFrame({'age':dataset.age,"is_adult":dataset.is_adult,"weight":dataset.weight})

### initial data visulization

def initial_vis(x,y):
    fig, ax = plt.subplots()
    ax.plot(x,y,'o')
    

initial_vis(dataset.age, dataset.weight)
plt.ylabel("weight(lb)", fontsize=18)
plt.xlabel("age(years)", fontsize=18)
plt.title("Initial Data Visulization(weight&age)")
plt.show()

initial_vis(dataset.weight,dataset.is_adult)
plt.xlabel("weight(lb)", fontsize=18)
plt.ylabel("ADULT=1 CHILD=0)", fontsize=18)
plt.title("Initial Data Visulization(is_adult&weight)")
plt.show()

initial_vis(dataset.age,dataset.is_adult)
plt.xlabel("age(years)", fontsize=18)
plt.ylabel("ADULT=1 CHILD=0)", fontsize=18)
plt.title("Initial Data Visulization(is_adult&age)")
plt.show()


###################
###Linear model
###################


### Since it rquires to have age<18,suset the dataframe then do the partition

# Age<18

df_linear=df.loc[df['age'] <18]

# Dataset partition
train_linear, test_linear = train_test_split(df_linear, test_size=0.2)

### Here we have age as x, weight as y
x_linear_train= np.array(train_linear['age'])
y_linear_train= np.array(train_linear['weight'])

x_linear_test= np.array(test_linear['age'])
y_linear_test= np.array(test_linear['weight'])

# Normalize the data



scaler.fit(x_linear_train.reshape(-1, 1))
x_linear_train_scale=(scaler.transform(x_linear_train.reshape(-1, 1)))

scaler.fit(x_linear_test.reshape(-1, 1))
x_linear_test_scale=(scaler.transform(x_linear_test.reshape(-1, 1)))

scaler.fit(y_linear_test.reshape(-1, 1))
y_linear_test_scale=(scaler.transform(y_linear_test.reshape(-1, 1)))

scaler.fit(y_linear_train.reshape(-1, 1))
y_linear_train_scale=(scaler.transform(y_linear_train.reshape(-1, 1)))

### Use SciPy optimizer to train the parameters

# use mean square error (MSE) as the loss function

def f_mse(m,b,x,y):
    y_pred=m*x+b
    mse=np.mean((y - y_pred)**2)
    return mse

# Use SciPy optimizer to train the parameters

#SAVE HISTORY FOR PLOTTING AT THE END


iterations=[]
loss_train=[]
loss_test=[]

iteration=0
def loss(p):
      global iteration,iterations,loss_train,loss_test
      training_loss=f_mse(p[0],p[1],x_linear_train_scale,y_linear_train_scale)
      test_loss=f_mse(p[0],p[1],x_linear_test_scale,y_linear_test_scale)
      loss_train.append(training_loss)
      loss_test.append(test_loss)
      iterations.append(iteration)

      iteration+=1
      
      return(training_loss)


linear_opt = minimize(loss, np.random.uniform(0.5,1,size=2), tol=1e-12)

# Prediction

y_pred_scale=linear_opt.x[1]+linear_opt.x[0]*x_linear_train_scale


### Unnormalized the data and Invert the fitting model to predict weight(age).
y_pred_inverse=(np.std(y_linear_train))*y_pred_scale+np.mean(y_linear_train).tolist()
y_pred_inverse= [ item for elem in y_pred_inverse for item in elem]
# Train MSE
mse_train_linear=np.mean((y_linear_train - np.array(y_pred_inverse))**2)

# Train MAE
mae_train_linear=np.mean(abs(y_linear_train - np.array(y_pred_inverse)))
# Test MSE
y_pred_scale_test=linear_opt.x[1]+linear_opt.x[0]*x_linear_test_scale
y_pred_inverse_test=(np.std(y_linear_test))*y_pred_scale_test+np.mean(y_linear_test).tolist()
y_pred_inverse_test=[ item for elem in y_pred_inverse_test for item in elem]
mse_test_linear=np.mean((y_linear_test - np.array(y_pred_inverse_test))**2)
# Test MAE
mae_test_linear=np.mean(abs(y_linear_test - np.array(y_pred_inverse_test)))

### plot the result

fig, ax = plt.subplots()
ax.plot(x_linear_train, y_linear_train, 'o', label="Training Set")
ax.plot(x_linear_train, y_pred_inverse, '-', label="Model")
ax.plot(x_linear_test, y_linear_test, 'ro', label="Testing")

ax.legend()
plt.ylabel("weight(lb)", fontsize=18)
plt.xlabel("age(years)", fontsize=18)
plt.title("Linear regression (weight&age<18)", fontsize=18)
plt.show()


### Ploting the Training and Testing Loss



fig, ax = plt.subplots()
ax.plot(iterations, loss_train, 'o', label="Training Loss")
ax.plot(iterations, loss_test, 'o', label="Testing Loss")


ax.legend()
plt.ylabel("Loss", fontsize=18)
plt.xlabel("iterations", fontsize=18)
plt.title("Training and Testing Loss", fontsize=18)
plt.show()

#PARITY PLOTS
fig, ax = plt.subplots()
ax.plot(y_pred_inverse, y_linear_train, 'o', label='Training set')
ax.plot(y_pred_inverse_test,y_linear_test, 'o', label='Validation set')


plt.xlabel('y predicted', fontsize=18)
plt.ylabel('y data', fontsize=18)
plt.legend()
plt.show()



### Print MSE, MAE

print('Train MSE (Linear regression): {}'.format(mse_train_linear)) 
print('Train MAE (Linear regression): {}'.format(mae_train_linear))
print('Test MSE (Linear regression): {}'.format(mse_test_linear))
print('Test MAE (Linear regression): {}'.format(mae_test_linear))

###################################
###Logistic Regression(weight&age)
###################################


### do the partition, no need to subset this time, apply to both logistic regression


# Dataset partition
train_log, test_log = train_test_split(df, test_size=0.2)

### Here we have age as x, weight as y
x_log_train= np.array(train_log['age'])
y_log_train= np.array(train_log['weight'])

x_log_test= np.array(test_log['age'])
y_log_test= np.array(test_log['weight'])

# Normalize the data


scaler.fit(x_log_train.reshape(-1, 1))
x_log_train_scale=(scaler.transform(x_log_train.reshape(-1, 1)))

scaler.fit(y_log_train.reshape(-1, 1))
y_log_train_scale=(scaler.transform(y_log_train.reshape(-1, 1)))

scaler.fit(x_log_test.reshape(-1, 1))
x_log_test_scale=(scaler.transform(x_log_test.reshape(-1, 1)))

scaler.fit(y_log_test.reshape(-1, 1))
y_log_test_scale=(scaler.transform(y_log_test.reshape(-1, 1)))



### Use SciPy optimizer to train the parameters

# use mean square error (MSE) as the loss function

def f_mse_log(a,w,x0,s,x,y):
    y_pred=a/(1+np.exp(-(x-x0)/w))+s
    mse=np.mean((y - y_pred)**2)
    return mse

# Use SciPy optimizer to train the parameters

iterations=[]
loss_train=[]
loss_test=[]

iteration=0
def loss(p):
      global iteration,iterations,loss_train,loss_test
      training_loss=f_mse_log(p[0],p[1],p[2],p[3],x_log_train_scale,y_log_train_scale)
      test_loss=f_mse_log(p[0],p[1],p[2],p[3],x_log_test_scale,y_log_test_scale)
      loss_train.append(training_loss)
      loss_test.append(test_loss)
      iterations.append(iteration)

      iteration+=1
      
      return(training_loss)

log_opt = minimize(loss, x0=np.random.uniform(0.5,1,size=4), tol=1e-12)




#make prediction
y_pred_scale=log_opt.x[0]/(1+np.exp(-(x_log_train_scale-log_opt.x[2])/log_opt.x[1]))+log_opt.x[3]
y_pred_inverse=(np.std(y_log_train))*y_pred_scale+np.mean(y_log_train).tolist()
y_pred_inverse= [ item for elem in y_pred_inverse for item in elem]



# Train MSE
mse_train_log=np.mean((y_log_train - np.array(y_pred_inverse))**2)

# Train MAE
mae_train_log=np.mean(abs(y_log_train - np.array(y_pred_inverse)))
# Test MSE
y_pred_scale_test=log_opt.x[0]/(1+np.exp(-(x_log_test_scale-log_opt.x[2])/log_opt.x[1]))+log_opt.x[3]
y_pred_inverse_test=(np.std(y_log_test))*y_pred_scale_test+np.mean(y_log_test).tolist()
y_pred_inverse_test=[ item for elem in y_pred_inverse_test for item in elem]
mse_test_log=np.mean((y_log_test - np.array(y_pred_inverse_test))**2)
# Test MAE
mae_test_log=np.mean(abs(y_log_test - np.array(y_pred_inverse_test)))

### plot the result
xs, ys = zip(*sorted(zip(x_log_train, y_pred_inverse)))
fig, ax = plt.subplots()
ax.plot(x_log_train, y_log_train, 'o',label="Training Set")
ax.plot(x_log_test, y_log_test, 'x',label="Testing Set")
ax.plot(xs, ys, 'r-',label="Model")

ax.legend()
plt.ylabel("weight(lb)", fontsize=18)
plt.xlabel("age(years)", fontsize=18)
plt.title("Logistic regression (weight&age)", fontsize=18)
plt.show()

### Ploting the Training and Testing Loss



fig, ax = plt.subplots()
ax.plot(iterations, loss_train, 'o', label="Training Loss")
ax.plot(iterations, loss_test, 'o', label="Testing Loss")


ax.legend()
plt.ylabel("Loss", fontsize=18)
plt.xlabel("iterations", fontsize=18)
plt.title("Training and Testing Loss", fontsize=18)
plt.show()


#PARITY PLOTS
fig, ax = plt.subplots()
ax.plot(y_pred_inverse, y_log_train, 'o', label='Training set')
ax.plot(y_pred_inverse_test,y_log_test, 'o', label='Validation set')
# ax.plot(yt, yt, '-', label='y_pred=y_data')

plt.xlabel('y predicted', fontsize=18)
plt.ylabel('y data', fontsize=18)
plt.legend()
plt.show()

### Print MSE, MAE

print('Train MSE (Logistic regression (weight&age)): {}'.format(mse_train_log)) 
print('Train MAE (Logistic regression (weight&age)): {}'.format(mae_train_log))
print('Test MSE (Logistic regression (weight&age)): {}'.format(mse_test_log))
print('Test MAE (Logistic regression (weight&age)): {}'.format(mae_test_log))


#########################################
###Logistic Regression(is_adult&weight)
#########################################

# We have done the partition in the last model


### Here we have weight as x, is_adult as y
y_log_train= np.array(train_log['is_adult'])
x_log_train= np.array(train_log['weight'])

y_log_test= np.array(test_log['is_adult'])
x_log_test= np.array(test_log['weight'])

# Normalize the x


scaler.fit(x_log_train.reshape(-1, 1))
x_log_train_scale=(scaler.transform(x_log_train.reshape(-1, 1)))



scaler.fit(x_log_test.reshape(-1, 1))
x_log_test_scale=(scaler.transform(x_log_test.reshape(-1, 1)))

x_log_train_scale=x_log_train_scale.tolist()
x_log_train_scale= [ item for elem in x_log_train_scale for item in elem]

### Use SciPy optimizer to train the parameters


# Use SciPy optimizer to train the parameters

def loss(p):
      global iteration,iterations,loss_train,loss_test
      training_loss=f_mse_log(p[0],p[1],p[2],p[3],x_log_train_scale,y_log_train)
      test_loss=f_mse_log(p[0],p[1],p[2],p[3],x_log_test_scale,y_log_test)
      loss_train.append(training_loss)
      loss_test.append(test_loss)
      iterations.append(iteration)

      iteration+=1
      
      return(training_loss)
  
log_opt = minimize(loss, x0=np.random.uniform(0.5,1,size=4), tol=1e-12)




y_pred=log_opt.x[0]/(1+np.exp(-(x_log_train_scale-log_opt.x[2])/log_opt.x[1]))+log_opt.x[3]



# Train MSE
mse_train_log=np.mean((y_log_train - y_pred)**2)

# Train MAE
mae_train_log=np.mean(abs(y_log_train - y_pred))
# Test MSE
y_pred_test=log_opt.x[0]/(1+np.exp(-(x_log_test_scale-log_opt.x[2])/log_opt.x[1]))+log_opt.x[3]
mse_test_log=np.mean((y_log_test - y_pred_test)**2)
# Test MAE
mae_test_log=np.mean(abs(y_log_test - y_pred_test))

# ### plot the result
xs, ys = zip(*sorted(zip(x_log_train, y_pred)))
fig, ax = plt.subplots()


ax.plot(x_log_train, y_log_train, 'o',label="Training Set")
ax.plot(x_log_test, y_log_test, 'x',label="Testing Set")
ax.plot(xs, ys, 'r-',label="Model")


ax.legend()
plt.ylabel("weight(lb)", fontsize=18)
plt.xlabel("age(years)", fontsize=18)
plt.title("Logistic regression (is_adult&weight)", fontsize=18)
plt.show()


### Ploting the Training and Testing Loss



fig, ax = plt.subplots()
ax.plot(iterations, loss_train, 'o', label="Training Loss")
ax.plot(iterations, loss_test, 'o', label="Testing Loss")


ax.legend()
plt.ylabel("Loss", fontsize=18)
plt.xlabel("iterations", fontsize=18)
plt.title("Training and Testing Loss", fontsize=18)
plt.show()



#PARITY PLOTS
fig, ax = plt.subplots()
ax.plot(y_pred, y_log_train, 'o', label='Training set')
ax.plot(y_pred_test,y_log_test, 'o', label='Validation set')
# ax.plot(yt, yt, '-', label='y_pred=y_data')

plt.xlabel('y predicted', fontsize=18)
plt.ylabel('y data', fontsize=18)
plt.legend()
plt.show()

### Print MSE, MAE

print('Train MSE (Logistic regression (is_adult&weight)): {}'.format(mse_train_log)) 
print('Train MAE (Logistic regression (is_adult&weight)): {}'.format(mae_train_log))
print('Test MSE (Logistic regression (is_adult&weight)): {}'.format(mse_test_log))
print('Test MAE (Logistic regression (is_adult&weight)): {}'.format(mae_test_log))

