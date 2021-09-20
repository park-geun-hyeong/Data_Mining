from itertools import combinations
from IPython.display import display
import time
import csv

def confidence(a, b, frequent_item_dict):
    support_b = frequent_item_dict[b]
    union = 0

    for i in data:
        if (a in i) and (b in i):
            union += 1

    return union / support_b


def association_rule_mining(data:list, support_th=100, confidence_th=0.2):
    
    
    def zero():
        return int(0)

    item_dict = defaultdict(zero)
    
    ## 1 pass    
    for i in data:
        for j in i:
            item_dict[j] += 1

    frequent_item_dict = dict()
    for i in sorted(item_dict.items(), key=lambda x: x[1], reverse=True):
        if i[1] > support_th:
            frequent_item_dict[i[0]] = i[1]
    
    frequent_item_key = [i for i in frequent_item_dict.keys()]

    combination = list(combinations(frequent_item_key, 2))
    # combination = [[frequent_item_key[i], frequent_item_key[j]] for i in range(len(frequent_item_key)) for j in range(i+1, len(frequent_item_key))]
 

    ## 2 pass 
    confidence_dict={}
    for i in combination_2:
        if confidence(i[0], i[1], frequent_item_dict) > confidence_th:
            confidence_dict[i] = confidence(i[0],i[1],frequent_item_dict)
            
    interest_dict = dict()
    for i in [i for i in sorted(confidence_dict.items(), key= lambda x: x[1], reverse=True)]:
        interest_dict[i[0]] = round((i[1] - frequent_item_dict[i[0][1]] / len(data)), 6)

    return interest_dict

if __name__ =='__main__':

    start = time.time()

    data = list()
    f = open('./store_data.csv', "r")
    rea = csv.reader(f)

    for i in rea:
        data.append(i)

    f.close()

    display(association_rule_mining(data, 30, 0.2))
    print(f"Time : {time.time() - start :.4f} sec, Num : {len(association_rule_mining(data, 30, 0.2))} ")
