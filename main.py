import sys
import csv
import numpy
import pandas
import matplotlib.pyplot as pyplot
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


train_data = pandas.read_csv("data/lsmCompe_train.csv",header=None)
original_train_data = pandas.read_csv("data/lsmCompe_train.csv",header=None)
test_data = pandas.read_csv("data/lsmCompe_test.csv",header=None)

"""
sum_x[n] == sum(x^n)
sum_xy[n] == sum(x^n * y)
"""

outlier_train_data = []

outlier_detect_blocksize = 100

#for i in range(0, len(train_data), outlier_detect_blocksize):
for i in range(0, len(train_data), 100):
    tmp_train_data = train_data.iloc[i:i+outlier_detect_blocksize,1]

    avg = tmp_train_data.mean()
    std = tmp_train_data.std()

    min_value = avg - std*2
    max_value = avg + std*2

    tmp_train_data.loc[ tmp_train_data < min_value] = None
    tmp_train_data.loc[ tmp_train_data > max_value] = None

train_data = train_data.dropna()

train_data, mytest_data = train_test_split(train_data, test_size=0.1)

train_data = train_data.loc[:,:].to_numpy(dtype=object)
original_train_data = original_train_data.loc[:,:].to_numpy(dtype=object)
mytest_data = mytest_data.loc[:,:].to_numpy(dtype=object)
#original_train_data = original_train_data.loc[:,:].to_numpy(dtype=numpy.int64)

#print(abs((train_data[:,1] - avg)/std))

#train_data = train_data[ abs((train_data[:,1] - avg)/std) < 2 ]
#outlier_train_data = train_data[ abs((train_data[:,1] - avg)/std) >= 2 ]

#print(outlier_train_data)

#print(train_data[:,0])

def learning2(train_index, test_index, lam, dimensionN):
    param = numpy.polyfit(train_data[train_index,0].tolist(),train_data[train_index,1].tolist(),dimensionN)
    param = numpy.matrix([[i] for i in param])
    return param

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

    for i in range(0,dimensionN):
        left_matrix[i][i] += lam

    for i in range(dimensionN + 1):
        right_matrix.insert(j, [sum_xy[dimensionN - i]])

    left_matrix = numpy.matrix(left_matrix)
    right_matrix = numpy.matrix(right_matrix)

    #w = left_matrix.I @ right_matrix
    w = numpy.linalg.solve(left_matrix,right_matrix)
    return w

def RMSE(mytest_data,w):
    mytrain_output = []

    for i in range(len(mytest_data)):
        y = f(mytest_data[i][0], w)
        mytrain_output.append(y.item())

    #print(mytest_data[:,1])
    #print(mytrain_output)
    return numpy.sqrt(mean_squared_error(mytest_data[:,1],mytrain_output))

def f(x,w):
    return sum(numpy.matrix(numpy.flip(numpy.logspace(0, len(w)-1, len(w), base=x))) * w)

def plotw(w,option):

    frange = numpy.arange(0,train_data[:,0].max()+5,1)

    ans = []

    for i in frange:
        ans.append( f(i,w).item() )

    pyplot.plot(frange,ans,option)


def run_sklearning(dimensionN,k_closs_validation,lam,show):
    if k_closs_validation == 0:
        global_w = [learning(range(len(train_data)), 0, lam, dimensionN)]
        global_rmse = [-1]
    else:
        global_w = []
        global_rmse = []
        index = 0;
        skf = KFold(n_splits=k_closs_validation)
        #skf = ShuffleSplit(n_splits=k_closs_validation, random_state=None)
        for train_index, test_index in skf.split(train_data[:,0],train_data[:,1]):
            sys.stdout.write("%3d%%\r" % (index * 100 / len(train_data)))
            sys.stdout.flush()
            
            #w = learning2(train_index, test_index, lam, dimensionN)
            w = learning(train_index, test_index, lam, dimensionN)
            rmse = RMSE(train_data[test_index],w)
            global_w.insert(index, w)
            global_rmse.insert(index, rmse)
            index+=1
            if show == True:
                plotw(w,'--')

        print("100%")

    return (global_w,global_rmse)

def get_average_rmse_for_kcv(dimensionN, k_closs_validation, lam, show):
    (global_w, global_rmse) = run_sklearning(dimensionN, k_closs_validation,lam,False)

    best_index = numpy.array(global_rmse).argmin()
    rmse = global_rmse[best_index]
    w = global_w[best_index]
    if show == True:
        plotw(w,'')
    print("average rmse:",numpy.average(global_rmse))
    return (w,rmse)

def best_lam(dimensionN, k_closs_validation, show):
    best_lam = 0
    best_rmse = 1000
    for i in range(0,20):
        print("lam:", -i)
        (w,rmse) = get_average_rmse_for_kcv(dimensionN, k_closs_validation,numpy.exp(-i), show)
        if rmse < best_rmse:
            best_rmse = rmse
            best_lam = -i
    return best_lam

def best_dimensionN(k_closs_validation, lam, show):
    best_dimensionN = 0
    best_rmse = 1000
    for i in range(1,20):
        (w,rmse) = get_average_rmse_for_kcv(i, k_closs_validation, lam, show)
        if rmse < best_rmse:
            best_rmse = rmse
            best_dimensionN = i
    return best_dimensionN

#dimensionN = 6
#dimensionN = 9
dimensionN = 20
k_closs_validation = len(train_data)
#k_closs_validation = 4
#k_closs_validation = 0
#lam = numpy.exp(-5)
lam = 0


(w,rmse) = get_average_rmse_for_kcv(dimensionN, k_closs_validation,lam, True)
plotw(w,'')
print(w)
print("final test rmse: ", RMSE(mytest_data,w))

#print(best_lam(dimensionN,k_closs_validation,False))
#print(best_dimensionN(k_closs_validation, lam, False))

pyplot.plot(original_train_data[:,0],original_train_data[:,1],'go')
pyplot.plot(train_data[:,0],train_data[:,1],'ro')
pyplot.plot(mytest_data[:,0],mytest_data[:,1],'bo')
pyplot.show()

"""
with open("./w.csv", 'a') as file1:
    writer = csv.writer(file1)
    writer.writerow(numpy.array(w).flatten())


with open("./w.csv") as file2:
    reader = csv.reader(file2, quoting=csv.QUOTE_NONNUMERIC)
    data = [row for row in reader]

data = numpy.array(data)
w = []
for i in range(dimensionN + 1):
        w.insert(i,[sum(data[:,i]) / len(data)])

print(w)
pyplot.plot(train_data[:,0],train_data[:,1],'ro')
pyplot.plot(mytest_data[:,0],mytest_data[:,1],'go')
plotw(w,'')

rmse = RMSE(mytest_data,w)
print(rmse)
pyplot.show()
"""

max_lam = 10
average_sample = 10
max_dimension = 20
max_k_closs = 15
