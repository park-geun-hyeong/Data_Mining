from collections import defaultdict
import time
import csv


def confidence(a, b, frequent_item_dict):
    support_a = frequent_item_dict[a]
    union = 0

    for i in data:
        if (a in i) and (b in i):
            union += 1

    return union / support_a


def association_rule_mining(data: list, support_th=10, confidence_th=0.1):
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

    combination = []
    for i in range(len(item_key)):
        for j in range(len(item_key)):
            if i == j:
                continue
            else:
                combination.append((item_key[i], item_key[j]))

    ## 2 pass
    confidence_dict = {}
    for i in combination:
        if confidence(i[0], i[1], frequent_item_dict) > confidence_th:
            confidence_dict[i] = confidence(i[0], i[1], frequent_item_dict)

    interest_dict = dict()
    for i in confidence_dict.items():
        interest_dict[f"{i[0][0]} ==> {i[0][1]}"] = round((i[1] - frequent_item_dict[i[0][1]] / len(data)), 6)

    return sorted(interest_dict.items(), key=lambda x: x[1], reverse=True)


if __name__ == '__main__':

    start = time.time()

    data = list()
    f = open('./store_data.csv', "r")
    rea = csv.reader(f)

    for i in rea:
        data.append(set(i))

    f.close()

    print("====== Interest of association rule =====")
    print(association_rule_mining(data)[:100])
    print(f"Time : {time.time() - start :.4f} sec, Num of itemSet : {len(association_rule_mining(data))}")