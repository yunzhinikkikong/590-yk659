#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 13:03:42 2021

@author: Nikkikong
"""


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
from   scipy.optimize import minimize

#USER PARAMETERS
IPLOT=True
INPUT_FILE='LectureCodes/weight.json'
FILE_TYPE="json"
DATA_KEYS=['x','is_adult','y']

# Repeat the logistic regression training for age vs weight
model_type="logistic"; NFIT=4; xcol=1; ycol=2;


#READ FILE
with open(INPUT_FILE) as f:
	my_input = json.load(f)  #read into dictionary

#CONVERT INPUT INTO ONE LARGE MATRIX 
X=[];
for key in my_input.keys():
	if(key in DATA_KEYS):
		X.append(my_input[key])

#MAKE ROWS=SAMPLE DIMENSION (TRANSPOSE)
#SIMILAR TO A PD DATAFRAME
X=np.transpose(np.array(X))

#SELECT COLUMNS FOR TRAINING 
x=X[:,xcol];  y=X[:,ycol]


#COMPUTE BEFORE PARTITION AND SAVE FOR LATER
XMEAN=np.mean(x); XSTD=np.std(x)
YMEAN=np.mean(y); YSTD=np.std(y)

#NORMALIZE
x=(x-XMEAN)/XSTD; y=(y-YMEAN)/YSTD; 

#PARTITION
f_train=0.8; f_val=0.2
rand_indices = np.random.permutation(x.shape[0])
CUT1=int(f_train*x.shape[0]); 
train_idx,  val_idx = rand_indices[:CUT1], rand_indices[CUT1:]

xt=x[train_idx]; yt=y[train_idx]
xv=x[val_idx];   yv=y[val_idx]

#MODEL
def model(x,p):
	
	return  p[0]+p[1]*(1.0/(1.0+np.exp(-(x-p[2])/(p[3]+0.00001))))

#SAVE HISTORY FOR PLOTTING AT THE END
iterations=[]; loss_train=[];  loss_val=[]
iteration=0
def loss(p, xm,ym):
	#TRAINING LOSS
	yp=model(xm,p) #model predictions for given parameterization p
	mse=(np.mean((yp-ym)**2.0))  #MSE
	return mse


### initial value
po=np.random.uniform(0.1,1.,size=NFIT)


def optimizer(obj,method,LR=0.1,algo = "MOM",tol=10**-8):
    
    
    #PARAM
    dx=0.001							#STEP SIZE FOR FINITE DIFFERENCE								#LEARNING RATE
    t=0 	 							#INITIAL ITERATION COUNTER
    tmax=5000		#MAX NUMBER OF ITERATION
			#EXIT AFTER CHANGE IN F IS LESS THAN THIS 
    xi=po
    NDIM=4
    left = 0
    alpha=0.7 # exponential decay factor in MOM
    if method == "batch":
        batchsize=len(xt)
    elif method == "mini batch":
        batchsize=100
    elif method == "SGD":
        batchsize = 1
        tmax=100000
    
    right = batchsize
    loss_train = []
    loss_val = []
  
    df_dx_pre = np.zeros(NDIM)
    while(t<tmax):
        t=t+1
        xsub = xt[left:right]
        ysub = yt[left:right]
            
    #NUMERICALLY COMPUTE GRADIENT 
        df_dx=np.zeros(NDIM)
        xi_loss = loss(xi,xsub,ysub)
        import copy
        for i in range(0,NDIM):
            dX=np.zeros(NDIM);
            dX[i]=dx; 
            xm1=xi-dX; #print(xi,xm1,dX,dX.shape,xi.shape)
            
            df_dx[i]=(xi_loss - loss(xm1, xsub,ysub))/dx
	#print(xi.shape,df_dx.shape)
    
        if algo == "GD":
            xip1=xi-LR*df_dx #STEP 
        elif algo == "MOM":
            xip1=xi-LR*df_dx-alpha*df_dx_pre*LR
            df_dx_pre = copy.deepcopy(df_dx)
        loss_train.append(xi_loss)
        loss_val.append(loss(xi,xv,yv))
        if(t%10==0):
            df=np.mean(np.absolute(loss(xip1,xsub,ysub)-loss(xi,xsub,ysub)))
            print(t,"	",xi,"	","	",loss(xi,xsub,ysub)) #,df) 

            if(df<tol):
                print("STOPPING CRITERION MET (STOPPING TRAINING)")
                break

	#UPDATE FOR NEXT ITERATION OF LOOP
        xi=xip1
        left = (left + batchsize) % len(xt)
        right = left + batchsize
        
    return (xi,loss_train,loss_val)
    
    
res,loss_train,loss_val = optimizer(loss,"batch",algo="MOM")
iterations=np.arange(len(loss_train))+1
popt=res
print("OPTIMAL PARAM:",popt)

#PREDICTIONS
xm=np.array(sorted(xt))
yp=np.array(model(xm,popt))

#UN-NORMALIZE
def unnorm_x(x): 
	return XSTD*x+XMEAN  
def unnorm_y(y): 
	return YSTD*y+YMEAN 


#FUNCTION PLOTS
if(IPLOT):
	fig, ax = plt.subplots()
	ax.plot(unnorm_x(xt), unnorm_y(yt), 'o', label='Training set')
	ax.plot(unnorm_x(xv), unnorm_y(yv), 'x', label='Validation set')
	ax.plot(unnorm_x(xm),unnorm_y(yp), 'r-', label='Model')
	plt.xlabel('x', fontsize=18)
	plt.ylabel('y', fontsize=18)
	plt.legend()
	plt.show()

#PARITY PLOTS
if(IPLOT):
	fig, ax = plt.subplots()
	ax.plot(model(xt,popt), yt, 'o', label='Training set')
	ax.plot(model(xv,popt), yv, 'o', label='Validation set')
	# ax.plot(yt, yt, '-', label='y_pred=y_data')

	plt.xlabel('y predicted', fontsize=18)
	plt.ylabel('y data', fontsize=18)
	plt.legend()
	plt.show()

#MONITOR TRAINING AND VALIDATION LOSS  

fig.clear(True)
if(IPLOT):
 	fig, ax = plt.subplots()
 	#iterations,loss_train,loss_val
 	ax.plot(iterations, loss_train, 'o', label='Training loss')
 	ax.plot(iterations, loss_val, 'o', label='Validation loss')
 	plt.xlabel('optimizer iterations', fontsize=18)
 	plt.ylabel('loss', fontsize=18)
 	plt.legend()
 	plt.show()
 
     
 
    
