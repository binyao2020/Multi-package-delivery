''' 
generate 3 random instances with 100 orders and uncertain customers
All customers are generated in a 2D planes [-50,50]*[-50,50], where depot locates at (0,0)
extract medium instances with 50 orders by considering the first 50 orders in the large instance
extract small instances with 25 orders by considering the first 25 orders in the large instance
return a dictionary for each instance: {customer_label, location, [size of all orders]}
'''

import numpy as np
import matplotlib as plt

n = 3 # number of instances (large and small)
for i in range(n):
    size_list = np.random.randint(5,high=16,size=100) # list of sizes of 100 items
    order_list = np.random.randint(0,high=50,size=100) # list of customer-order matching of 100 items
    loc_list = 100*np.random.rand(50,2)-50 # location
    order_dict = {}
    label = 0 
    for k in range(50):
        if list(order_list).count(k) != 0:
            order = []
            for j in range(100):
                if order_list[j] == k:
                    order.append(size_list[j])
            order_dict[label] = [loc_list[k][0],loc_list[k][1],order]
            label += 1
    np.save(str(i+1)+'_L.npy',order_dict) 
    count = 0
    l = 0
    while 1>0:
        if count+len(order_dict[l][2]) < 50:
            count += len(order_dict[l][2])
            l += 1
        else:
            residual = 50-count
            order_dict[l][2] = order_dict[l][2][:residual]
            for k in range(l+1,label):
                order_dict.pop(k)
            break
    np.save(str(i+1)+'_M.npy',order_dict) 
    count = 0
    label = len(order_dict.keys())
    l = 0
    while 1>0:
        if count+len(order_dict[l][2]) < 25:
            count += len(order_dict[l][2])
            l += 1
        else:
            residual = 25-count
            order_dict[l][2] = order_dict[l][2][:residual]
            for k in range(l+1,label):
                order_dict.pop(k)
            break
    np.save(str(i+1)+'_S.npy',order_dict) 