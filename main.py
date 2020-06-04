import numpy
import pandas
import matplotlib.pyplot as pyplot
from sklearn.model_selection import StratifiedKFold

dimensionN = 6
k_closs_validation = 3

skf = StratifiedKFold(n_splits=k_closs_validation)

train_data = pandas.read_csv("data/lsmCompe_train.csv",header=None)
original_train_data = train_data
test_data = pandas.read_csv("data/lsmCompe_test.csv",header=None)

"""
sum_x[n] == sum(x^n)
sum_xy[n] == sum(x^n * y)
"""

avg = train_data.mean()[1]
std = train_data.std()[1]

train_data = train_data.loc[:,:].to_numpy(dtype=object)
original_train_data = original_train_data.loc[:,:].to_numpy(dtype=numpy.int64)

#print(abs((train_data[:,1] - avg)/std))

#train_data = train_data[ abs((train_data[:,1] - avg)/std) < 2 ]
outlier_train_data = train_data[ abs((train_data[:,1] - avg)/std) >= 2 ]

#print(outlier_train_data)

print(train_data[:,0])
global_w = []
index = 0;
for train_index, test_index in skf.split(original_train_data[:,0],original_train_data[:,1]):
    #print("train: ", train_index, " test: ", test_index)
    #train_data = train_data[train_index,0]

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

    for i in range(dimensionN + 1):
        right_matrix.insert(j, [sum_xy[dimensionN - i]])

    left_matrix = numpy.matrix(left_matrix)
    right_matrix = numpy.matrix(right_matrix)

    print("left_matrix=", left_matrix)
    print("right_matrix=", right_matrix)
    w = left_matrix.I @ right_matrix
    #print("w=", w)
    global_w.insert(index, w)

    def f(x):
        return numpy.matrix(numpy.flip(numpy.logspace(0, dimensionN, dimensionN + 1, base=x))) * w

#pyplot.plot(original_train_data[:,0],original_train_data[:,1],'ro')
    pyplot.plot(train_data[train_index,0],train_data[train_index,1],'ro')
    #pyplot.plot(outlier_train_data[train_index,0],outlier_train_data[train_index,1],'bo')
    pyplot.plot(train_data[test_index,0],train_data[test_index,1],'go')

    frange = numpy.arange(0,original_train_data[:,0].max(),1)

    ans = []

    for i in frange:
        ans.append( f(i).item() )

    pyplot.plot(frange,ans)

    #pyplot.show()
    index+=1

print(global_w)
final_w = []
for i in range(dimensionN + 1):
    final_w.insert(i, 0)
    for j in range(k_closs_validation):
        final_w[i] += global_w[j][i]
    final_w[i] /= k_closs_validation

print(final_w)
