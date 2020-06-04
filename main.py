import numpy
import pandas
import matplotlib.pyplot as pyplot

dimensionN = 4

train_data = pandas.read_csv("data/lsmCompe_train.csv",header=None)
test_data = pandas.read_csv("data/lsmCompe_test.csv",header=None)

"""
sum_x[n] == sum(x^n)
sum_xy[n] == sum(x^n * y)
"""

sum_x = numpy.array(None)
sum_xy = []

for i in range(dimensionN * 2 + 1):
    sum_x = numpy.insert(sum_x, i, sum(train_data.loc[:,0].values ** i))
    #sum_x.insert(i, sum(train_data.loc[:,0].values ** i))

for i in range(dimensionN + 1):
    sum_xy.insert(i, sum((train_data.loc[:,0].values ** i) * train_data.loc[:,1].values))

#sum_x2 = sum(train_data.loc[:,0].values ** 2)
sum_y = sum(train_data.loc[:,1].values)
#sum_xy = sum(train_data.loc[:,0].values * train_data.loc[:,1].values)
sum_1 = train_data.index.stop

#print(sum_1)
#print(sum_x)
#print(sum_x2)
#print(sum_y)
#print(sum_xy)

left_matrix = []
right_matrix = []

for i in range(dimensionN + 1):
    item = []
    for j in range(dimensionN + 1):
        item.insert(j, sum_x[2*dimensionN - i - j])
    left_matrix.insert(i, item)

for i in range(dimensionN + 1):
    right_matrix.insert(j, [sum_xy[dimensionN - i]])

left_matrix = numpy.matrix(left_matrix)
right_matrix = numpy.matrix(right_matrix)

print(left_matrix)
print(right_matrix)
w = left_matrix.I @ right_matrix
#(a,b) = (a.item(),b.item())

print(numpy.logspace(0, dimensionN, dimensionN + 1, base=3))

def f(x):
    return numpy.matrix(numpy.flip(numpy.logspace(0, dimensionN, dimensionN + 1, base=x))) * w

pyplot.plot(train_data.loc[:,0],train_data.loc[:,1],'ro')

frange = numpy.arange(0,train_data.max()[0],1)

ans = []

for i in frange:
    ans.append( f(i).item() )

pyplot.plot(frange,ans)

pyplot.show()
