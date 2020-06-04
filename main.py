import numpy
import pandas
import matplotlib.pyplot as pyplot

train_data = pandas.read_csv("data/lsmCompe_train.csv",header=None)
test_data = pandas.read_csv("data/lsmCompe_test.csv",header=None)

sum_x = sum(train_data.loc[:,0].values)
sum_x2 = sum(train_data.loc[:,0].values ** 2)
sum_y = sum(train_data.loc[:,1].values)
sum_xy = sum(train_data.loc[:,0].values * train_data.loc[:,1].values)
sum_1 = train_data.index.stop

print(sum_1)
print(sum_x)
print(sum_x2)
print(sum_y)
print(sum_xy)

left_matrix = numpy.matrix([[sum_x2, sum_x]
                           ,[sum_x,  sum_1]])
right_matrix = numpy.matrix([[sum_xy],
                             [sum_y]])

(a,b) = left_matrix.I @ right_matrix
(a,b) = (a.item(),b.item())

def f(x):
    return a*x + b

pyplot.plot(train_data.loc[:,0],train_data.loc[:,1],'ro')
frange = numpy.arange(0,train_data.max()[0],1)
pyplot.plot(frange,f(frange))
pyplot.show()
