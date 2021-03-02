"""
This code is for method 1 where we first generate packing solution then provide routing solution with TSP
"""

from gurobipy import *
import csv
import numpy as np
from math import *
from itertools import product
import time as tm
import matplotlib as plt

def BFD(Q,size_list): # use BFD heuristics to provide upper bound of # of vehicles
    s = sorted(size_list,reverse=True)
    count = 1
    remain = [Q]
    for i in range(len(s)):
        insert = -1
        min_remain = inf
        for j in range(count):
            if remain[j]>=s[i] and remain[j]<min_remain:
                insert = j
                min_remain = remain[j]
        if insert != -1:
            remain[insert] -= s[i]
        else:
            count += 1
            remain.append(Q-s[i])
    return count
    
def dist_mat(order_dict): # generate distance matrix for all pairs of nodes
    l = len(order_dict.keys())
    D = np.zeros((l,l))
    for i in range(l):
        for j in range(i+1,l):
            D[i][j] = sqrt((order_dict[i][0]-order_dict[j][0])**2+(order_dict[i][1]-order_dict[j][1])**2)
            D[j][i] = D[i][j]
    return D        

def packing(order_dict,size_list,index_list,Q,U,obj): # provide packing solution by solving an IP model
    customers = range(len(order_dict.keys())-1)
    orders = range(len(size_list))
    vehicles = range(U)
    BP = Model("bin-packing") 
    BP.setParam("OutputFlag",0)
    x = BP.addVars(list(product(orders,vehicles)),vtype=GRB.BINARY, name="x") # order-vehicle
    y = BP.addVars(U,vtype=GRB.BINARY, name="y") # vehicle
    z = BP.addVars(list(product(customers,vehicles)),vtype=GRB.BINARY, name="z") # customer-vehicle
    if obj == 0:
        BP.setObjective(quicksum(y[j] for j in vehicles))
    else:
        BP.setObjective(quicksum(z[c,j] for (c,j) in list(product(customers,vehicles)))+quicksum(y[j] for j in vehicles))
    for i in orders:
        BP.addConstr(quicksum(x[i,j] for j in range(U)) == 1)
    for j in vehicles:
        BP.addConstr(quicksum(size_list[i]*x[i,j] for i in orders) <= Q*y[j])
    for (c,j) in list(product(customers,vehicles)):
        BP.addConstr(quicksum(x[i,j] for i in index_list[c]) <= len(index_list[c])*z[c,j])
    BP.optimize()
    ysol = BP.getAttr('x', y)
    xsol = BP.getAttr('x', x)
    V = sum(ysol)
    pack_info = []
    for j in vehicles:
        pack = []
        for i in orders:
            if xsol[i,j] == 1:
                for k in range(len(index_list)):
                    if i in index_list[k]:
                        pack.append([k,size_list[i]])
                        continue
        if pack != []:
            pack_info.append(pack)
    return V,pack_info
    
def routing(D,order_info): # provide routing soln using TSP for each clusters
    node_list = list(set([order_info[i][0] for i in range(len(order_info))]))+[len(D)-1]
    n = len(node_list)
    nodes = range(n)
    pairs = list(product(nodes,nodes))
    for j in nodes:
        pairs.remove((j,j))    
    TSP = Model("TSP")
    TSP.setParam("OutputFlag",0)
    x = TSP.addVars(pairs,vtype=GRB.BINARY, name="x")
    u = TSP.addVars(n,ub=n-1,vtype=GRB.INTEGER, name="u")
    TSP.setObjective(quicksum(D[node_list[i]][node_list[j]]*x[i,j] for (i,j) in pairs))
    for j in nodes: # degree
        TSP.addConstr(quicksum(x[i,j] for i in set(nodes)-{j}) == 1)
        TSP.addConstr(quicksum(x[j,i] for i in set(nodes)-{j}) == 1)
    TSP.addConstr(u[n-1] == 0)
    for k in set(nodes)-{n-1}:
        for j in set(nodes)-{k}:
            TSP.addConstr(u[k]>=1+u[j]-n*(1-x[j,k]))
    TSP.optimize()
    miles = TSP.ObjVal
    route = [len(D)-1]
    xsol = TSP.getAttr('x',x)
    succ = n-1
    while 1>0:
        curr = succ
        for j in set(nodes)-{curr}:
            if xsol[curr,j] == 1:
                route.append(node_list[j])
                succ = j
                break
        if succ == n-1:
            break                    
    return route,miles
    
def fleet_size(total_miles,mile_list,route_list): # provide final fleet_size by grouping single-trip route into multi-trip route
    U = BFD(total_miles,mile_list)
    trips = range(len(mile_list))
    vehicles = range(U)
    FS = Model("fleet size")
    FS.setParam("OutputFlag",0)
    x = FS.addVars(list(product(trips,vehicles)),vtype=GRB.BINARY, name="x") # trip-vehicle
    y = FS.addVars(U,vtype=GRB.BINARY, name="y") # vehicle
    FS.setObjective(quicksum(y[j] for j in vehicles))
    for i in trips:
        FS.addConstr(quicksum(x[i,j] for j in vehicles) == 1)
    for j in vehicles:
        FS.addConstr(quicksum(mile_list[i]*x[i,j] for i in trips) <= total_miles*y[j])
    FS.optimize()
    V = FS.ObjVal # final fleet size
    ysol = FS.getAttr('x', y)
    xsol = FS.getAttr('x', x)
    final_route_list = []
    for j in vehicles:
        if ysol[j] == 1:
            multi_trip = [route_list[0][0]]
            for i in trips:
                if xsol[i,j] == 1:
                    multi_trip += route_list[i][1:]
            final_route_list.append(multi_trip)
    return V, final_route_list

if __name__=='__main__':  
    start_time = tm.time()
    Q = 60
    order_dict = np.load('1_S.npy',allow_pickle=True).item() 
    order_dict[len(order_dict.keys())] = [0,0,[]] #information about depot
    D = dist_mat(order_dict)
    size_list = []
    index_list = []
    count = 0
    for key in set(order_dict.keys())-{len(order_dict.keys())-1}:
        index = []
        for k in order_dict[key][2]:
            size_list.append(k)
            index.append(count)
            count += 1
        index_list.append(index)  
    U = BFD(Q,size_list)
    V0,packing_soln = packing(order_dict,size_list,index_list,Q,U,0)
    V1,packing_soln = packing(order_dict,size_list,index_list,Q,U,1)
    if V0 != V1:
        print('W!','\n')
    route_list = []
    mile_list = []
    for order_info in packing_soln:
        route, miles = routing(D,order_info)
        route_list.append(route)
        mile_list.append(miles)
    total_miles = 440
    V, final_route_list = fleet_size(total_miles,mile_list,route_list)
    print("# of vehicles = ",V,"\n")  
    print("final schedule = ",final_route_list,"\n")
    print("total miles = ",sum(mile_list),"\n")
    end_time = tm.time() 
    print("p1r2_Time = "+str(end_time-start_time),"\n")


  