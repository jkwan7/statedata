
# coding: utf-8

# In[1]:

# The Data Incubator Challenge Question 3
# Jeffrey Kwan

import openpyxl
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:

# Open California excel sheet
California = openpyxl.load_workbook('California.xlsx')


# In[3]:

employment = np.zeros((3,53))
education = np.zeros((8,53))
sector = np.zeros((14,53))

for i in range(0,53):
    sheet = California.get_sheet_by_name('District ' + str(i+1))
    # Total civilian labor force; number employed; number unemployed
    for j in range(0,3):
        employment[j][i] = sheet['B' + str(j+103)].value
    # Total population 25 years and over; less than 9th grade; 9th to 12th grade, no diploma...
    # high school graduate (includes equivalency); some college, no degree; associate's degree...
    # bachelor's degree; graduate or professional degree; % high school graduate or higher; $ bachelor's degree or higher
    for j in range(0,8):
        education[j][i] = sheet['B' + str(j+262)].value
    # Civilian employed population 16 years and over; Agriculture, forestry, fishing and hunting, and mining;
    # Construction, Manufacturing, Wholesale trade; Retail trade; Transportation and warehousing, and utilities;
    # Information; Finance and insurance, and real estate and rental and leasing
    # Professional, scientific, and management, and administrative and waste management services; 
    # Educational services, and health care and social assistance; Arts, entertainment, and recreation, and accommodation and food services;
    # Other services, except public administration; Public administration
    for j in range(0,14):
        sector[j][i] = sheet['B' + str(j+127)].value
    


# In[4]:

employmentper = employment[1][:]/employment[0][:]

educationper = education[1:][:]/education[0][:]
highplus = np.ones((1,53))-educationper[0][:]-educationper[1][:]
bachelorplus = np.expand_dims(educationper[5][:]+educationper[6][:],axis=0)
gradprof = np.expand_dims(educationper[6][:],axis=0)

sectorper = sector[1:][:]/sector[0][:]
secdiversity = np.zeros((1,53))
d = 13 # Industries
# Normalize based on L2 norm
for i in range(0,53):
    secdiversity[0,i] = (np.linalg.norm(sectorper[:,i])*np.sqrt(d)-1)/(np.sqrt(d)-1)


# In[5]:

y = np.matrix(np.expand_dims(employmentper,axis=0))
X = np.matrix(np.r_[np.ones((1,53)),highplus,bachelorplus,gradprof,secdiversity])
theta0 = np.matrix(np.zeros((1,5)))
iterations = 200
alpha = 0.01

J_history = np.ones((1,iterations))*np.inf

def costfunc(X,y,theta):
    m = len(y)
    J = (1/(2*m))*np.sum((np.square(theta*X-y)))
    return(J)

J_history[0][0] = costfunc(X,y,theta0)


def ode_gradient(theta,alpha,iterations):
    m = len(y)
    for i in range(0,iterations):
        thetastep = theta
        for j in range(0,theta.shape[1]):
            theta.put(j,np.take(theta,j)-alpha*(1/m)*np.sum(np.multiply((thetastep*X-y),X[:][j])))
        J_history[0][i]=costfunc(X,y,theta)
    return(theta)

theta = ode_gradient(theta0,alpha,iterations)


# In[6]:

districts = range(1,54)
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(districts,employment[0,:],'r')
ax2.plot(districts,employmentper,'b')
ax1.set_xlabel('District Number')
ax1.set_ylabel('Total Labor Force',color='r')
ax2.set_ylabel('Percentage Employed',color='b')
ax1.set_xlim(0,55)
plt.title('Labor Force')

plt.show()


# In[7]:

# Open Texas excel sheet
Texas = openpyxl.load_workbook('Texas.xlsx')


# In[8]:

employment2 = np.zeros((3,36))
education2 = np.zeros((8,36))
sector2 = np.zeros((14,36))

for i in range(0,36):
    sheet2 = Texas.get_sheet_by_name('District ' + str(i+1))
    for j in range(0,3):
        employment2[j][i] = sheet2['B' + str(j+103)].value
    for j in range(0,8):
        education2[j][i] = sheet2['B' + str(j+262)].value
    for j in range(0,14):
        sector2[j][i] = sheet2['B' + str(j+127)].value


# In[9]:

employmentper2 = employment2[1][:]/employment2[0][:]

educationper2 = education2[1:][:]/education2[0][:]
highplus2 = np.ones((1,36))-educationper2[0][:]-educationper2[1][:]
bachelorplus2 = np.expand_dims(educationper2[5][:]+educationper2[6][:],axis=0)
gradprof2 = np.expand_dims(educationper2[6][:],axis=0)

sectorper2 = sector2[1:][:]/sector2[0][:]
secdiversity2 = np.zeros((1,36))
d = 13 # Industries
# Normalize based on L2 norm
for i in range(0,36):
    secdiversity2[0,i] = (np.linalg.norm(sectorper2[:,i])*np.sqrt(d)-1)/(np.sqrt(d)-1)


# In[10]:

realy = employmentper
realy2 = employmentper2
X2 = np.matrix(np.r_[np.ones((1,36)),highplus2,bachelorplus2,gradprof2,secdiversity2])
predictedy = np.squeeze(np.array(theta*X))
predictedy2 = np.squeeze(np.array(theta*X2))


# In[19]:

districts2 = range(1,37)

plt.figure(figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(districts,realy-predictedy,label='California Real - Predicted Employment Percentage')
plt.plot(districts2,realy2-predictedy2,label='Texas Real - Predicted Employment Percentage')
plt.plot(range(0,56),np.zeros(56))
plt.xlabel('District Number')
plt.ylabel('Employment Percentage')
plt.legend(loc='best')
plt.xlim(0,55)
plt.show()


# In[ ]:



