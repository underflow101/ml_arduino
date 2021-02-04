import os
from micromlgen import port
from sklearn.svm import SVC
from sklearn.datasets import load_iris

def mkdir(dir, k):
    if not os.path.exists("./" + dir + str(k)):
        os.mkdir("./" + dir + str(k))
        return dir + str(k)
    else:
        mkdir(dir, k+1)

def write_file(dir, model):
    f = open("./" + dir + "/model.h", 'w')
    f.write(model)
    f.close()

if __name__ == '__main__':
    iris = load_iris()
    x = iris.data
    y = iris.target
    clf = SVC(kernel='linear', gamma=0.001).fit(x, y)
    model = port(clf)
    
    dir = mkdir("model", 0)
    write_file(dir, model)