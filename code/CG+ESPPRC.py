"""
This code is for method 2 where we generate pack and route solution using column generation with ESPPRC algorithm for subproblem
"""

from gurobipy import *
import csv
import numpy as np
from math import *
from itertools import product
import time as tm
import matplotlib.pyplot as plt

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

def multi_to_single(order_dict): # decompose the original dictinary into the one only contains single order
    single_order_dict = {}
    label = 0
    for i in range(len(order_dict.keys())):
        for j in range(len(order_dict[i][2])):
            single_order_dict[label] = [order_dict[i][0],order_dict[i][1],order_dict[i][2][j]]
            label += 1
    return single_order_dict
    
def dist_mat(order_dict): # generate distance matrix for all pairs of nodes
    l = len(order_dict.keys())
    D = np.zeros((l,l))
    for i in range(l):
        for j in range(i+1,l):
            D[i][j] = sqrt((order_dict[i][0]-order_dict[j][0])**2+(order_dict[i][1]-order_dict[j][1])**2)
            D[j][i] = D[i][j]
    return D  

def CWS(D,size_list): # the Clarke-Wright savings heuristic to generate initial columns for column generation
    cws_start_time = tm.time()
    savings = []
    n = len(D)-1 # index of depot
    for i in range(n):
        for j in set(range(i+1,n)):
            savings.append([i,j,D[n][i]+D[n][j]-D[i][j]])
    savings.sort(reverse=True,key=lambda x:x[2])
    route_list = [[i] for i in range(n)] #initial: each order forms a route
    cap_list = [size_list[i] for i in range(n)]
    for sav in savings:
        i = sav[0]
        j = sav[1]
        flag_i = False
        flag_j = False
        for k in range(len(route_list)):
            if i == route_list[k][0] or i == route_list[k][-1]:
                index_i = k
                flag_i = True
                break
        for k in range(len(route_list)):
            if j == route_list[k][0] or j == route_list[k][-1]:
                index_j = k
                flag_j = True
                break
        if flag_i == True and flag_j == True and cap_list[index_i]+cap_list[index_j] <= Q and index_i != index_j:
            route_list[index_i] += route_list[index_j]
            del route_list[index_j]
            cap_list[index_i] += cap_list[index_j]
            del cap_list[index_j]
    mile_list = []
    for route in route_list:
        miles = D[n][route[0]]+D[route[-1]][n]+sum(D[route[k]][route[k+1]] for k in range(len(route)-1))
        mile_list.append(miles)
    V,_ = fleet_size(440,mile_list,route_list)
    print('cws_V',V,'\n')
    cws_end_time = tm.time()
    print('cws_time', cws_end_time-cws_start_time,'\n')
    print('cws_fleet',len(route_list),'\n')
    print('cws_miles',sum(mile_list),'\n')
    return route_list,mile_list

def dominance(label,label2): # examine the dominance relationship between label and label2
    state = [int(k) for k in label[0]]
    state2 = [int(k) for k in label2[0]]
    gap = [state[i]-state2[i] for i in range(len(state))]
    if max(gap) <= 0 and label[1] <= label2[1] and label[2] <= label2[2] and (state != state2 or label[1] != label2[1] or label[2] != label2[2]):
        return 1 # new dominates old
    if min(gap) >= 0 and label[1] >= label2[1] and label[2] >= label2[2]:
        return 2 # old dominates new or the same
    else:
        return 0 # no dominance

def ESPPRC(D,size_list,profit): #Exact algorithm for ESPPRC to solve the subproblem
    #print('profit',profit,'\n')
    obj = profit.pop(-1)
    n = len(D) # n-1: end depot; n: start depot
    ################### generate modified distance matrix #######################
    D2 = np.zeros((n+1,n+1)) # the last two are depot and dummy depot
    D2[n][n-1] = np.inf
    profit += 0,0
    size_list += 0,0
    for j in set(range(n+1))-{n-1,n}:
        D2[n][j] = D[n-1][j]-0.5*profit[j]
    for i in range(n-1):
        D2[i][n-1] = D[i][n-1]-0.5*profit[i]
    for i in range(n-1): # exclude dummy depot
        for j in range(i+1,n-1): # exclude depot and i itself
            D2[i][j] = D[i][j]-0.5*profit[i]-0.5*profit[j]
            D2[j][i] = D2[i][j]
    ################### Initiallization: define label set for each node #########################
    label_list = [[] for i in range(n+1)]
    label_list[n].append([[0 for i in range(n)],0,0]) #[[status of all nodes], #used quantity, cost(distance)]
    queue = [n]
    #print('s',size_list,'\n')
    #################### algorithm ########################
    espprc_start_time = tm.time()
    while queue != []:
        curr = queue.pop(0)
        for succ in set(range(n))-{curr}:
            L = [] # store the label extends from curr to succ
            for label in label_list[curr]:
                if label[0][succ] == 0 and label[1]+size_list[succ] <= Q: # feasible extension
                    surrogate = [label[0][i] for i in range(n)]
                    new_label = [label[i] for i in range(1,len(label))]
                    new_label.insert(0,surrogate)
                    new_label[1] += size_list[succ]
                    new_label[2] += D2[curr][succ]
                    new_label[0][succ] = 1
                    for i in range(n):
                        if i == 0 and new_label[1]+size_list[i] > Q:
                            new_label[0][i] = 1.5
                    L.append(new_label)
            if label_list[succ] == []:
                label_list[succ] = L
                if succ not in queue:
                    queue.append(succ)
            else:
                flag = False
                count = 0
                for i in range(len(L)):
                    label = L[count]
                    count2 = 0
                    for j in range(len(label_list[succ])):
                        label2 = label_list[succ][count2]
                        judge = dominance(label,label2)
                        if judge == 1:
                            flag = True
                            label_list[succ].pop(count2)
                        elif judge == 2:
                            L.pop(count)
                            count -= 1
                            break
                        elif judge == 0:
                            count2 += 1
                            if j == len(label_list[succ])-1:
                                flag = True
                    count += 1
                if flag == True:
                    if succ not in queue:
                        queue.append(succ)
                    for new_label in L:
                        label_list[succ].append(new_label)
        count = 0
        for label in label_list[n-1]:
            if label[2] < obj:
                count += 1
        espprc_end_time = tm.time()
        if count >= 10 or (espprc_end_time-start_time > 600): # stop criteria
            if count == 0:
                return [],[]
            column_list = []
            obj_list = []
            for label in label_list[n-1]:
                if label[2] < obj:
                    col = [0]*(n-1)
                    obj_pra = label[2]
                    for k in range(n-1):
                        if label[0][k] == 1:
                            col[k] = 1
                            obj_pra += profit[k]
                    column_list.append(col)
                    obj_list.append(obj_pra)
            return column_list, obj_list
    '''
    if count > 0:
        print(count,'\n')
        column_list = []
        obj_list = []
        for label in label_list[n-1]:
            if label[2] < obj:
                col = [0]*(n-1)
                obj_pra = label[2]
                for k in range(n-1):
                    if label[0][k] == 1:
                        col[k] = 1
                        obj_pra += profit[k]
                column_list.append(col)
                obj_list.append(obj_pra)
        return column_list, obj_list
    '''
    return [],[]
    
def col_gen(D,size_list): # column generation
    master = Model("pack and route")
    master.setParam("OutputFlag",0)
    route_list, mile_list = CWS(D,size_list) # use clarke-wright saving heuristics to generate initial routes
    m = len(D)-1 # index of depot or number of orders
    A = []
    for route in route_list:
        s = [0]*m
        for i in route:
            s[i] = 1
        A.append(s)
    U = len(A)
    order = {}
    x = {}
    for j in range(len(route_list)):
        x[j] = master.addVar(obj=mile_list[j],vtype=GRB.CONTINUOUS)
    for i in range(m):
        order[i] = master.addConstr(quicksum(A[j][i]*x[j] for j in range(len(route_list))) >= 1)
    UB = master.addConstr(quicksum(x[j] for j in range(len(route_list))) <= U)
    num_var = len(route_list)
    master.update()
    ite = 0
    
    while 1:
        master.optimize()
        print(ite,master.ObjVal,'\n\n')
        if master.ObjVal < 441:
            break
        profit = [order[i].Pi for i in range(m)] + [UB.Pi]
        column_list,obj_list = ESPPRC(D,size_list,profit) # use ESPPRC algorithm to solve the subproblem
        if column_list != []:
            for column in column_list:
                column = [int(column[i]) for i in range(m)]
                if column not in A:
                    A.append(column)
                    mile_list.append(obj_list.pop(0))
                    col = Column()
                    for i in range(m):
                        col.addTerms(column[i],order[i])
                    col.addTerms(1,UB)
                    x[num_var] = master.addVar(obj=mile_list[-1],vtype=GRB.CONTINUOUS,column=col)
                    num_var += 1 
            master.update() 
            ite += 1
            if tm.time()-start_time > 600:
                print('10 minutes TIME OUT!')
                break
        else:
            break
    
    for i in range(num_var):
        x[i].setAttr("vtype", GRB.INTEGER)
    master.update()
    master.optimize()
    final_route_list = []
    final_mile_list = []
    xsol = master.getAttr('x',x)
    for i in range(num_var):
        if xsol[i] == 1:
            final_route_list.append(A[i])
            final_mile_list.append(mile_list[i])
    return final_route_list,final_mile_list
    
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
    order_dict = multi_to_single(order_dict)
    size_list = [order_dict[i][2] for i in range(len(order_dict.keys()))]
    order_dict[len(order_dict.keys())] = [0,0,[]] #information about depot
    D = dist_mat(order_dict)
    route_list,mile_list = col_gen(D,size_list)
    total_miles = 440
    V, final_route_list = fleet_size(total_miles,mile_list,route_list)
    print("# of vehicles = ",V,"\n")  
    print("final schedule = ",final_route_list,"\n")
    print("total miles = ",sum(mile_list),"\n")
    end_time = tm.time() 
    print("p&r_Time = "+str(end_time-start_time),"\n")

    
   