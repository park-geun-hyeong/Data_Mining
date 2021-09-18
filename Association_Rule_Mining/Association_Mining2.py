import pandas as pd
import numpy as np
from itertools import combinations
from IPython.display import display
from collections import defaultdict
import time
import csv

def association_rule_mining(data:list, support_th=100, confidence_th=0.2):

    def zero():
        return int(0)

    item_dict = defaultdict(zero)

    for i in data:
        for j in i:
            item_dict[j] += 1

    frequent_item = [i for i in sorted(item_dict.items(), key=lambda x: x[1], reverse=True) if i[1] > support_th]
    frequent_item_key = [i[0] for i in frequent_item]

    frequent_item_dict = dict()
    for i in frequent_item:
        frequent_item_dict[i[0]] = i[1]

    combination = list(combinations(frequent_item_key, 2))
    ## combination = [[frequent_item_key[i], frequent_item_key[j]] for i in range(len(frequent_item_key)) for j in range(i+1, len(frequent_item_key))]

    combination_support = defaultdict(zero)
    for i in combination:
        for j in data:
            if (i[0] in j) and (i[1] in j):
                combination_support[i] += 1

    confidence_dict = defaultdict(zero)
    for i in combination:
        confidence_dict[i] = combination_support[i] / frequent_item_dict[i[1]]

    interest_dict = defaultdict(zero)
    for i in [i for i in confidence_dict.items() if i[1] > confidence_th]:
        interest_dict[i[0]] = round((i[1] - frequent_item_dict[i[0][1]] / len(data)), 6)

    key = [i[0] for i in sorted(interest_dict.items(), key=lambda x: x[1], reverse=True)]
    value = [i[1] for i in sorted(interest_dict.items(), key=lambda x: x[1], reverse=True)]

    Df = pd.DataFrame([i for i in zip(key, value)], columns = ['itemset', 'interest'])

    return Df

if __name__ =='__main__':

    start = time.time()

    data = list()
    f = open('./store_data.csv', "r")
    rea = csv.reader(f)

    for i in rea:
        data.append(i)

    f.close()

    display(association_rule_mining(data, 50, 0.22))
    print(f"time : {time.time() - start :.4f})