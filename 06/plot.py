#! /usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
from math import log


reducers = pd.DataFrame(
    columns=['CoreSeconds', 'Elapsed', 'AverageMapTime', 'AverageShuffleTime', 'AverageMergeTime', 'AverageReduceTime'])

def parse_time(string):
    out = 0
    for val in string.split(','):
        val = val.strip()
        if val.endswith('mins'):
            out += 60 * int(val[:-len('mins')])
        elif val.endswith('sec'):
            out += int(val[:-len('sec')])
        else:
            raise ValueError('cant parse' + string)
    return out


with open('r') as rfile:
    raw_data = rfile.read().strip().split('\n\n')
    for reducer in raw_data:
        rows = reducer.split('\n')
        name = rows[0]
        tmp_dict = dict()
        for row in rows[1:]:
            c_name, value = row.split(': ')
            tmp_dict[c_name] = value
        s = pd.Series(tmp_dict)
        s.name = int(name)
        reducers = reducers.append(s)

ucolumns = ['Elapsed', 'AverageMapTime', 'AverageShuffleTime', 'AverageMergeTime', 'AverageReduceTime']
for c_name in ucolumns:
    reducers[c_name] = reducers[c_name].apply(parse_time)
reducers['CoreSeconds'] = reducers['CoreSeconds'].apply(int)

for c_name in ucolumns:
    column = reducers[c_name]
    plt.plot(column.index, column.values, label=c_name)


plt.title('Зависимость времени выполнения от количества редьюсеров')
plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)
plt.xlabel('количество редьюсеров')
plt.savefig('report/times.png', bbox_inches="tight")
plt.show()

for c_name in ucolumns:
    column = reducers[c_name]
    plt.plot(column.index, column.values, label=c_name)

plt.yscale('log')
plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)
plt.xlabel('количество редьюсеров')
plt.savefig('report/times-log.png', bbox_inches="tight")
plt.show()


plt.title('Зависимость CoreSeconds от количества редьюсеров')
plt.plot(reducers.index, reducers['CoreSeconds'].values, label='CoreSeconds')
plt.legend()
plt.xlabel('количество редьюсеров')
plt.savefig('report/core-seconds.png', bbox_inches="tight")
plt.show()

