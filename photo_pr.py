# 学習データとテスト用画像から数字を識別する

import csv
import pprint
import numpy as np


number = 7 #　テスト用画像 1から8で入力
           #  1,5 -> 1, 2,6 -> 2 3,7 -> 3 4,8 -> 4 のテスト用画像


def step_function(x): # 0より大きい時はそのまま、0より小さい時は0を返す
    for i in range(len(x)):
        if x[i] < 0:
            x[i] = 0
    return x

weight1_w0 = np.loadtxt('weight1_w0.csv', delimiter=',') # 1層目のW0 10行
weight1 = np.loadtxt('weight1.csv', delimiter=',') # 1層目のW 10×64の行列
weight2_w0 = np.loadtxt('weight2_w0.csv', delimiter=',') # 2層目のW0　4行
weight2 = np.loadtxt('weight2.csv', delimiter=',') # 2層目のW　4×10の行列
test_data = np.loadtxt('test_photo_2.csv', delimiter=',') #　テスト用画像 63列

a1 = np.dot(weight1 , test_data[number - 1]) + weight1_w0 #　1層目の重みとテスト用画像の行列の積
z1 = step_function(a1) # 関数を使う
a2 = np.dot(weight2, z1) + weight2_w0 # 2層目の重みとz1の積
y = np.argmax(a2) + 1 # 数字を識別する

print(y) #　識別した数字を表示
