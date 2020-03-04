import csv
import pprint
import numpy as np


number = 7 #　テスト用画像 1から8で入力


def step_function(x):
    for i in range(len(x)):
        if x[i] < 0:
            x[i] = 0
    return x

weight1_w0 = np.loadtxt('weight1_w0.csv', delimiter=',')
weight1 = np.loadtxt('weight1.csv', delimiter=',')
weight2_w0 = np.loadtxt('weight2_w0.csv', delimiter=',')
weight2 = np.loadtxt('weight2.csv', delimiter=',')
test_data = np.loadtxt('test_photo_2.csv', delimiter=',')

a1 = np.dot(weight1 , test_data[number - 1]) + weight1_w0
z1 = step_function(a1)
a2 = np.dot(weight2, z1) + weight2_w0
y = np.argmax(a2) + 1

print(y)
