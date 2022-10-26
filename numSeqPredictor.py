import csv
import numpy as np
from sklearn.linear_model import LinearRegression
import re
import matplotlib.pyplot as plt


def get_past_seqs():
    f = open('files/numbers7.csv', 'r')
    file_read = csv.reader(f)
    array = list(file_read)
    num_array = [list(map(int, i)) for i in array]
    f.close()
    return num_array


past_num_seqs = get_past_seqs()
print(past_num_seqs)

x=[]
for i in range(0, len(past_num_seqs)):
    x.append([i])

to_predict_x = [len(past_num_seqs)+1]
to_predict_x = np.array(to_predict_x).reshape(-1, 1)
print(to_predict_x)

regsr = LinearRegression()
regsr.fit(x, past_num_seqs)

predicted_y = regsr.predict(to_predict_x)
m = regsr.coef_
c = regsr.intercept_
print("Predicted y:\n", predicted_y)
print("slope (m): ", m)
print("y-intercept (c): ", c)
