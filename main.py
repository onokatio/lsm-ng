import numpy
import pandas
import matplotlib.pyplot as pyplot
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


train_data = pandas.read_csv("data/lsmCompe_train.csv",header=None)
#original_train_data = train_data
train_data, mytest_data = train_test_split(train_data, test_size=0.25)
test_data = pandas.read_csv("data/lsmCompe_test.csv",header=None)

"""
sum_x[n] == sum(x^n)
sum_xy[n] == sum(x^n * y)
"""

avg = train_data.mean()[1]
std = train_data.std()[1]

train_data = train_data.loc[:,:].to_numpy(dtype=object)
mytest_data = mytest_data.loc[:,:].to_numpy(dtype=object)
#original_train_data = original_train_data.loc[:,:].to_numpy(dtype=numpy.int64)

#print(abs((train_data[:,1] - avg)/std))

#train_data = train_data[ abs((train_data[:,1] - avg)/std) < 2 ]
outlier_train_data = train_data[ abs((train_data[:,1] - avg)/std) >= 2 ]

#print(outlier_train_data)

#print(train_data[:,0])

def learning(train_index, test_index,lam, dimensionN):
    sum_x = []
    sum_xy = []

    for i in range(dimensionN * 2 + 1):
        sum_x = numpy.insert(sum_x, i, sum(numpy.power(train_data[train_index,0], i)))

    for i in range(dimensionN + 1):
        sum_xy = numpy.insert(sum_xy, i, sum(numpy.power(train_data[train_index,0], i) * train_data[train_index,1]))

    left_matrix = []
    right_matrix = []

    for i in range(dimensionN + 1):
        item = []
        for j in range(dimensionN + 1):
            item.insert(j, sum_x[2*dimensionN - i - j])
        left_matrix.insert(i, item)

    for i in range(2,dimensionN + 1):
        left_matrix[i][i] += lam

    for i in range(dimensionN + 1):
        right_matrix.insert(j, [sum_xy[dimensionN - i]])

    left_matrix = numpy.matrix(left_matrix)
    right_matrix = numpy.matrix(right_matrix)

    w = left_matrix.I @ right_matrix
    #global_w.insert(index, w)
    return w


    #pyplot.plot(train_data[train_index,0],train_data[train_index,1],'ro')

    ##pyplot.plot(outlier_train_data[train_index,0],outlier_train_data[train_index,1],'bo')
    ##pyplot.plot(train_data[test_index,0],train_data[test_index,1],'go')

    #frange = numpy.arange(0,original_train_data[:,0].max(),1)

    #ans = []

    #for i in frange:
    #    ans.append( f(i,w).item() )

    #pyplot.plot(frange,ans)

def f(x,w):
    return sum(numpy.matrix(numpy.flip(numpy.logspace(0, len(w)-1, len(w), base=x))) * w)

def run_sklearning(dimensionN,k_closs_validation):
    global_w = []
    index = 0;
    skf = KFold(n_splits=k_closs_validation, shuffle=True, random_state=None)
    for train_index, test_index in skf.split(train_data[:,0],train_data[:,1]):
        w = learning(train_index, test_index, 10 ** 0, dimensionN)
        global_w.insert(index, w)
        index+=1

    final_w = []
    for i in range(dimensionN + 1):
        final_w.insert(i, [0])
        for j in range(k_closs_validation):
            final_w[i][0] += global_w[j][i].item()
        final_w[i][0] /= k_closs_validation

    return final_w

def plotw(w):

    frange = numpy.arange(0,train_data[:,0].max(),1)

    ans = []

    for i in frange:
        ans.append( f(i,w).item() )

    pyplot.plot(train_data[:,0],train_data[:,1],'ro')
    pyplot.plot(frange,ans)

def RMSE(mytest_data,w):
    mytrain_output = []

    for i in range(len(mytest_data)):
        y = f(mytest_data[i][0], w)
        mytrain_output.append(y.item())

    print(mytest_data[:,1])
    print(mytrain_output)
    return numpy.sqrt(mean_squared_error(mytest_data[:,1],mytrain_output))

dimensionN = 10
k_closs_validation = 5

w = run_sklearning(dimensionN, k_closs_validation)
plotw(w)
print(w)

rmse = RMSE(mytest_data,w)
print("RMSE: ", rmse)
pyplot.show()
