import pandas as pd
import numpy as np
from itertools import combinations
from IPython.display import display
import time
import csv

def confidence(a, b, support_cut_dict):
    support_b = support_cut_dict[b]
    union = 0

    for i in data:
        if (a in i) and (b in i):
            union += 1

    return union / support_b


def interest(a, b, support_cut_dict):
    conf = confidence(a, b, support_cut_dict)
    p = support_cut_dict[b] / len(data)

    interest = conf - p
    return interest

def association_rule_mining(data:list, support_th=100, confidence_th=0.2):
    item_dict = dict()

    for i in data:
        for j in i:
            try:
                item_dict[j] += 1
            except KeyError:
                item_dict[j] = 1

    support_cut = []
    for i in item_dict.items():
        if i[1] > support_th:
            support_cut.append(i)

    support_cut_dict = dict()
    for i in support_cut:
        support_cut_dict[i[0]] = i[1]

    item_key = [i[0] for i in support_cut]
    combination_2 = list(combinations(item_key, 2))

    confidence_cut = [[a, b] for a, b in combination_2 if confidence(a, b, support_cut_dict) > confidence_th]
    Interest = [interest(i[0], i[1], support_cut_dict) for i in confidence_cut]
    interest_item = [i for i in zip(Interest, confidence_cut)]

    Df = pd.DataFrame(sorted(interest_item, reverse=True), columns=['interest', 'itemsets'])

    return Df

if __name__ =='__main__':

    start=time.time()

    data = list()
    f = open('./store_data.csv', "r")
    rea = csv.reader(f)

    for i in rea:
        data.append(i)

    f.close()

    display(association_rule_mining(data, 50, 0.22))
    print(f"time : {time.time() - start :.4f} sec")