from collections import defaultdict
from itertools import combinations
import time
import csv


def confidence(a, b, frequent_item_dict):
    '''
    Calculation confidence between item A and B

    :param a: name of item A
    :param b: name of item B
    :param frequent_item_dict: supports(dict) of frequent_items
    :return: confidence of item A ==> B & confidence of item B ==> A
    '''

    support_a = frequent_item_dict[a]
    support_b = frequent_item_dict[b]
    union = 0

    for i in data:
        if (a in i) and (b in i):
            union += 1

    return [union / support_a, union / support_b]


def association_rule_mining(data: list, support_th=100, confidence_th=0.2):
    '''
    AFriori Algorithm for association rule mining

    :param data: list of transaction
    :param support_th: support threshold(default : 100)
    :param confidence_th: confidence threshold(default : 0.2)
    :return: sorted list of Association rule's confidence & interest
    '''

    def zero():
        return int(0)

    item_support = defaultdict(zero)

    ## 1 pass
    for i in data:
        for j in i:
            item_support[j] += 1

    frequent_item_dict = dict()
    for i in sorted(item_support.items(), key=lambda x: x[1], reverse=True):
        if i[1] > support_th:
            frequent_item_dict[i[0]] = i[1]

    item_key = [i for i in frequent_item_dict.keys()]
    combination = list(combinations(item_key, 2))

    ## 2 pass
    confidence_dict = {}
    for i in combination:
        confidence_dict[(i[0], i[1])] = confidence(i[0], i[1], frequent_item_dict)[0]
        confidence_dict[(i[1], i[0])] = confidence(i[0], i[1], frequent_item_dict)[1]

    include_interest = [
        [f"{i[0][0]} ==> {i[0][1]}", round(i[1], 6), round((i[1] - frequent_item_dict[i[0][1]] / len(data)), 6)]
         for i in sorted(confidence_dict.items(), key=lambda x: x[1], reverse=True) if i[1] > confidence_th]

    return include_interest


if __name__ == '__main__':

    start = time.time()

    data = list()
    f = open('./store_data.csv', "r")
    rea = csv.reader(f)

    for i in rea:
        data.append(set(i))

    f.close()

    print("====== Confidence & Interest of association rule =====\n")
    for i in association_rule_mining(data):
        print(i)
    print(f"\nTime : {time.time() - start :.4f} sec, Num of ItemSet : {len(association_rule_mining(data))}")