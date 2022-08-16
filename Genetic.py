import random as rd
import pandas as pd
import matplotlib.pyplot as plt
import time

def draw_graph(iteration, fitness):
    plt.plot(iteration, fitness, 'g-', label='Genetic_value')
    plt.title('Graph showing the relationship between Iteration and Fitness_Value')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness Value')
    plt.legend(loc='best')
    plt.show()

def draw_graph_reality(x, y):
    color = ((rd.random(), rd.random(), rd.random()))
    plt.plot(x, y, linewidth='2',c=color, label='Genetic_value')

    plt.title('Graph showing the relationship between Iteration and Fitness_Value')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness Value')
    plt.legend(loc='best')
    plt.show()

def mutationRate_accepted(Sequence, childSequence, rate):
    temp = []

    for i in range(len(Sequence)):
        if Sequence[i] == childSequence[i]: 
            temp.append(Sequence[i])   

    percent = len(temp)/len(Sequence)
    if percent < rate: return False

    return True

def mutationRate_accepted_test(Sequence, childSequence):
    temp = []

    for i in range(len(Sequence)):
        if Sequence[i] == childSequence[i]: 
            temp.append(Sequence[i])   

    percent = len(temp)/len(Sequence)

    return percent

def cutPosition(job,number):
    ratio = int(job/number)

    cut_position = [0]*number

    for i in range(0, number):
        cut_position[i] = 0 + (i+1)*ratio

    if cut_position[number - 1] != job:
        cut_position[number - 1] = job

    return cut_position

def resetContain(cont):
    jobTime = {}

    for i in range(0, cont):
        jobTime.update({i+1 : 0})

    return jobTime

def Get_simulate_list(Trucks, Cranes, jobs):
    lst = []

    for job in range(1, jobs+1):
        for truck_num in range(0, len(Trucks)):
            if job in Trucks[truck_num]:
                truck_idx = truck_num+1
                break
        for crane_num in range(0, len(Cranes)):
            if job in Cranes[crane_num]:
                crane_idx = crane_num+1
                break
        lst.append([job, truck_idx, crane_idx])

    return lst

def Truck_mutation(count, n, trucks):
    s = 0
    truck_slot = n+trucks+1
    truck_seq = [0]*truck_slot
    p = []
    for i in range(0, len(truck_seq[0:n])):
        r = rd.sample(range(1,n+1), n)
        q = r[n-1]
        for j in range(n+1, truck_slot-1):
            values = r[0:n-1]
            randidx = rd.sample(range(len(values)), trucks-1)
            randidx.sort()
            b = randidx
    for k in range(len(b)):
        p.append(values[b[k]])
    truck_seq = [*r,s,*p,q]
    Truck_Seq = [*p,q]
    Trucksequence = [count, *truck_seq]
    cut_position = [0]*len(Truck_Seq)

    for i in range(0, trucks):
        cc1 = 2
        dd1 = 1
        while dd1 < n+2:
            if Trucksequence[n+i+2] == Trucksequence[cc1-1]:
                cut_position[i] = cc1-1
            cc1 = cc1 + 1
            dd1 = dd1 + 1
    # cut_position = cutPosition(n, trucks)

    Trucksequence = [count, *r, s, *cut_position]

    return Trucksequence, cut_position

def Crane_mutation(count, n, cranes):
    s = 0
    crane_slot = n+cranes+1
    crane_sq = [0]*crane_slot
    p = []
    for i in range(0, len(crane_sq[0:n])):
        r = rd.sample(range(1,n+1), n)
        q = r[n-1]
        for j in range(n+1, crane_slot-1):
            values = r[0:n-1]
            randidx = rd.sample(range(len(values)), cranes-1)
            randidx.sort()
            b = randidx
    for k in range(len(b)):
        p.append(values[b[k]])

    crane_sq = [*r,s,*p,q]
    Crane_Seq = [*p,q]
    Cranesequence = [count, *crane_sq]

    cut_position = [0]*len(Crane_Seq)

    for i in range(0, cranes):
        cc1 = 2
        dd1 = 1
        while dd1 < n+2:
            if Cranesequence[n+i+2] == Cranesequence[cc1-1]:
                cut_position[i] = cc1-1
            cc1 = cc1 + 1
            dd1 = dd1 + 1

    # cut_position = cutPosition(n, cranes)

    Cranesequence = [count, *r, s, *cut_position]

    return Cranesequence, cut_position

def Fitness_Caculate(ChildTruckSeq, Truck_cutPosition, ChildCraneSeq, Crane_cutPosition,
                     truck_ready, truck_processtime, crane_ready, crane_processtime):


    ######################################################################### DATA #############################################################

    Truck = [0]*len(Truck_cutPosition)
    Truck_ready = [0]*len(Truck_cutPosition)
    Truck_process = [0]*len(Truck_cutPosition)

    Crane = [0]*len(Crane_cutPosition)
    Crane_ready = [0]*len(Crane_cutPosition)
    Crane_process = [0]*len(Crane_cutPosition)

    jobFinishTime = resetContain(len(truck_ready))

    Truck[0] = ChildTruckSeq[1:Truck_cutPosition[0]+1]
    Truck[0].sort()

    for i in range(1, len(Truck_cutPosition)):
        Truck[i] = ChildTruckSeq[Truck_cutPosition[i-1]+1:Truck_cutPosition[i]+1]
        Truck[i].sort()

    Crane[0] = ChildCraneSeq[1:Crane_cutPosition[0]+1]
    Crane[0].sort()

    for i in range(1, len(Crane_cutPosition)):
        Crane[i] = ChildCraneSeq[Crane_cutPosition[i-1]+1:Crane_cutPosition[i]+1]
        Crane[i].sort()


    ############################################################ ASSIGN TRUCK READY ######################################################################

    for i in range(0, len(Truck_ready)):
        Truck_ready[i] = [0]*len(Truck[i])

    for k in range(0, len(Truck)):
        for j in range(len(truck_ready)):
            for i in range(len(Truck[k])):
                if Truck[k][i] == j+1:
                    Truck_ready[k][i] = truck_ready[j]

    ############################################################# ASSIGN TRUCK PROCESS ####################################################################

    for i in range(0, len(Truck_process)):
        Truck_process[i] = [0]*len(Truck[i])

    for k in range(0, len(Truck)):
        for j in range(len(truck_processtime)):
            for i in range(len(Truck[k])):
                if Truck[k][i] == j+1:
                    Truck_process[k][i] = truck_processtime[j]

    ###################################################################### TRUCK COMPLETION TIME ##########################################################
    Truck_Complete = [0]*len(Truck)

    for k in range(0, len(Truck)):
        Truck_Complete[k] = Truck_ready[k][0] + Truck_process[k][0]

        for x in jobFinishTime.keys():
            if x == Truck[k][0]:
                jobFinishTime[x] = jobFinishTime[x] + Truck_process[k][0]

        for i in range(1, len(Truck[k])):
            if Truck_ready[k][i] < Truck_Complete[k]:              
                for x in jobFinishTime.keys():
                    if x == Truck[k][i]:
                        jobFinishTime[x] = jobFinishTime[x] + Truck_process[k][i] + Truck_Complete[k] - Truck_ready[k][i]
                        
                Truck_Complete[k] = Truck_Complete[k] + Truck_process[k][i]
            else:
                for x in jobFinishTime.keys():
                    if x == Truck[k][i]:
                        jobFinishTime[x] = jobFinishTime[x] + Truck_process[k][i]
                Truck_Complete[k] = Truck_ready[k][i] + Truck_process[k][i]

    # Cmax_Truck = max(Truck_Complete)
    # Cmax_Truck = sum(Truck_Complete)


    ############################################################ ASSIGN CRANE READY ######################################################################

    for i in range(0, len(Crane_ready)):
        Crane_ready[i] = [0]*len(Crane[i])


    for k in range(0, len(Crane)):
        for j in range(len(crane_ready)):
            for i in range(len(Crane[k])):
                if Crane[k][i] == j+1:
                    Crane_ready[k][i] = crane_ready[j]

    ############################################################# ASSIGN CRANE PROCESS ####################################################################

    for i in range(0, len(Crane_process)):
        Crane_process[i] = [0]*len(Crane[i])

    for k in range(0, len(Crane)):
        for j in range(len(crane_processtime)):
            for i in range(len(Crane[k])):
                if Crane[k][i] == j+1:
                    Crane_process[k][i] = crane_processtime[j]

    ###################################################################### CRANE COMPLETION TIME ##########################################################
    Crane_Complete = [0]*len(Crane)

    for k in range(0, len(Crane)):
        
        # Ban đầu (tại cont đầu tiên) cont không phải chờ, cont tới lúc nào sẽ được crane chuyển đi lúc đó!
        for x in jobFinishTime.keys():
            if x == Crane[k][0]:
                jobFinishTime[x] = jobFinishTime[x] + Crane_process[k][0]
      
        Crane_Complete[k] = Crane_Complete[k] + Crane_process[k][0] 

        # Xét từ cont thứ 2 trở đi --->
        for i in range(1, len(Crane[k])):
            for x in jobFinishTime.keys():        
                if x == Crane[k][i]:

                    # Nếu cont sau đến TRƯỚC khi crane vận chuyển xong cont trước ---> cont sau phải chờ!
                    if jobFinishTime[x] < Crane_Complete[k]:            
                        wait_time = Crane_Complete[k] - jobFinishTime[x] # --> Thời gian cont sau chờ crane làm xong cont trước
                        jobFinishTime[x] = jobFinishTime[x] + Crane_process[k][i] + wait_time
                        Crane_Complete[k] = Crane_Complete[k] + Crane_process[k][i]

                    # Nếu cont sau đến SAU khi crane vận chuyển xong cont trước ---> cont sau không phải chờ!
                    else:
                        Crane_Complete[k] = jobFinishTime[x] + Crane_process[k][i]
                        jobFinishTime[x] = jobFinishTime[x] + Crane_process[k][i]

    Cmax_Crane = max(Crane_Complete)
    averangeTime = float(sum(jobFinishTime.values())/len(jobFinishTime.values()))
    # Cmax_Crane = sum(Crane_Complete)

    ###################################################################### CMAX OF SOLUTION ############################################################

    # Cmax = sum([Cmax_Truck, Cmax_Crane])

    return averangeTime, jobFinishTime.values()

def Genetic(n, a, c, truck_rate, crane_rate, truck_ready, truck_processtime, crane_ready, crane_processtime, result_file):

    #######################################################################################
    ######## GENERATE SEQUENCES RANDOMLY WITH n JOBs, a YTs, c YCs, b Yard blocks #######
    #######################################################################################

    start_time = time.time()

    truck_slot = n+a+1
    block_slot = n+c+1
    s = 0

    ###################################################################### Truck Sequence ####################################################################

    truck_sq = [0]*truck_slot
    p = []
    for i in range(0, len(truck_sq[0:n])):
        r = rd.sample(range(1,n+1), n)
        q = r[n-1]
        for j in range(n+1, truck_slot-1):
            values = r[0:n-1]
            randidx = rd.sample(range(len(values)), a-1)
            randidx.sort()
            b = randidx
    for k in range(len(b)):
        p.append(values[b[k]])
    truck_sq = [*r,s,*p,q]
    Truck_Seq = [*p,q]
    Trucksequence = [1, *truck_sq]
    cut_position = [0]*len(Truck_Seq)

    for i in range(0, a):
        cc1 = 2
        dd1 = 1
        while dd1 < n+2:
            if Trucksequence[n+i+2] == Trucksequence[cc1-1]:
                cut_position[i] = cc1-1
            cc1 = cc1 + 1
            dd1 = dd1 + 1

    # cut_position = cutPosition(n, a)

    Parent_truck_cut_position = cut_position

    Trucksequence = [0, *r, s, *cut_position]

    Truck = [0]*a
    Truck[0] = Trucksequence[1:cut_position[0]+1]
    Truck[0].sort()

    for i in range(1, a):
        Truck[i] = Trucksequence[cut_position[i-1]+1:cut_position[i]+1]
        Truck[i].sort()

    ############################################################### ####### BLOCK SEQUENCE ####################################################################

    block_sq = [0]*block_slot
    p = []
    for i in range(0, len(block_sq[0:n])):
        r = rd.sample(range(1,n+1), n)
        q = r[n-1]
        for j in range(n+1, block_slot - 1):
            values = r[0:n-1]
            randidx = rd.sample(range(len(values)), c-1)
            randidx.sort()
            b = randidx
    for k in range(len(b)):
        p.append(values[b[k]])
    block_sq = [*r,s,*p,q]
    Block_Seq = [*p,q]
    Blocksequence = [1, *block_sq]
    cut_position = [0]*len(Block_Seq)

    for i in range(0, c):
        cc1 = 2
        dd1 = 1
        while dd1 < n+2:
            if Blocksequence[n+i+2] == Blocksequence[cc1-1]:
                cut_position[i] = cc1-1
            cc1 = cc1 + 1
            dd1 = dd1 + 1
    
    # cut_position = cutPosition(n, c)

    Blocksequence = [0, *r, s, *cut_position]
    

    ###################################################################### CRANE SEQUENCE ##################################################################

    crane_sq = block_sq
    Crane_Seq = Block_Seq
    Cranesequence = [1, *crane_sq]

    cut_position = [0]*len(Crane_Seq)

    for i in range(0, c):
        cc1 = 2
        dd1 = 1
        while dd1 < n+2:
            if Cranesequence[n+i+2] == Cranesequence[cc1-1]:
                cut_position[i] = cc1-1
            cc1 = cc1 + 1
            dd1 = dd1 + 1
    
    # cut_position = cutPosition(n, c)

    Parent_crane_cut_position = cut_position

    Cranesequence = [0, *r, s, *cut_position]

    Crane = [0]*c
    Crane[0] = Cranesequence[1:cut_position[0]+1]
    Crane[0].sort()

    for i in range(1, c):
        Crane[i] = Cranesequence[cut_position[i-1]+1:cut_position[i]+1]
        Crane[i].sort()


    #######################################################################################################################################################
    ############################################################# COMPLETION TIME #########################################################################
    #######################################################################################################################################################

    ############################################################# ASSIGN TRUCK READY ######################################################################

    Truck_ready = [0]*len(Truck)
    Truck_process = [0]*len(Truck)

    jobFinishTime = resetContain(n)

    for i in range(0, len(Truck_ready)):
        Truck_ready[i] = [0]*len(Truck[i])

    for k in range(0, len(Truck)):
        for j in range(len(truck_ready)):
            for i in range(len(Truck[k])):
                if Truck[k][i] == j+1:
                    Truck_ready[k][i] = truck_ready[j]

    ############################################################# ASSIGN TRUCK PROCESS ####################################################################

    for i in range(0, len(Truck_process)):
        Truck_process[i] = [0]*len(Truck[i])

    for k in range(0, len(Truck)):
        for j in range(len(truck_processtime)):
            for i in range(len(Truck[k])):
                if Truck[k][i] == j+1:
                    Truck_process[k][i] = truck_processtime[j]

    ###################################################################### TRUCK COMPLETION TIME ##########################################################
    Truck_Complete = [0]*len(Truck)

    for k in range(0, len(Truck)):
        Truck_Complete[k] = Truck_ready[k][0] + Truck_process[k][0]

        for x in jobFinishTime.keys():
            if x == Truck[k][0]:
                jobFinishTime[x] = jobFinishTime[x] + Truck_process[k][0]

        for i in range(1, len(Truck[k])):
            if Truck_ready[k][i] < Truck_Complete[k]:      
                for x in jobFinishTime.keys():
                    if x == Truck[k][i]:
                        jobFinishTime[x] = jobFinishTime[x] + Truck_process[k][i] + Truck_Complete[k] - Truck_ready[k][i]
                Truck_Complete[k] = Truck_Complete[k] + Truck_process[k][i]
            else:          
                for x in jobFinishTime.keys():
                    if x == Truck[k][i]:
                        jobFinishTime[x] = jobFinishTime[x] + Truck_process[k][i]
                Truck_Complete[k] = Truck_ready[k][i] + Truck_process[k][i]

    ############################################################ ASSIGN CRANE READY ######################################################################
    Crane_ready = [0]*len(Crane)
    Crane_process = [0]*len(Crane)

    # Crane1 = [1, 3, 6], Crane2 = [2], Crane3 = [4]
    # Crane4 = [7, 8], Crane5 = [5, 10], Crane6 = [9]

    for i in range(0, len(Crane_ready)):
        Crane_ready[i] = [0]*len(Crane[i])


    for k in range(0, len(Crane)):
        for j in range(len(crane_ready)):
            for i in range(len(Crane[k])):
                if Crane[k][i] == j+1:
                    Crane_ready[k][i] = crane_ready[j]

    ############################################################# ASSIGN CRANE PROCESS ####################################################################

    for i in range(0, len(Crane_process)):
        Crane_process[i] = [0]*len(Crane[i])

    for k in range(0, len(Crane)):
        for j in range(len(crane_processtime)):
            for i in range(len(Crane[k])):
                if Crane[k][i] == j+1:
                    Crane_process[k][i] = crane_processtime[j]

    ###################################################################### CRANE COMPLETION TIME ##########################################################
    Crane_Complete = [0]*len(Crane)

    # Bắt đầu ở Crane 1 (k = 0)
    for k in range(0, len(Crane)):
        
        # Ban đầu (tại cont đầu tiên) cont không phải chờ, cont tới lúc nào sẽ được crane chuyển đi lúc đó!
        for x in jobFinishTime.keys():
            if x == Crane[k][0]:
                jobFinishTime[x] = jobFinishTime[x] + Crane_process[k][0]
      
        Crane_Complete[k] = Crane_Complete[k] + Crane_process[k][0] 

        # Xét từ cont thứ 2 trở đi --->
        for i in range(1, len(Crane[k])):
            for x in jobFinishTime.keys():        
                if x == Crane[k][i]:

                    # Nếu cont sau đến TRƯỚC khi crane vận chuyển xong cont trước ---> cont sau phải chờ!
                    if jobFinishTime[x] < Crane_Complete[k]:            
                        wait_time = Crane_Complete[k] - jobFinishTime[x] # --> Thời gian cont sau chờ crane làm xong cont trước
                        jobFinishTime[x] = jobFinishTime[x] + Crane_process[k][i] + wait_time
                        Crane_Complete[k] = Crane_Complete[k] + Crane_process[k][i]

                    # Nếu cont sau đến SAU khi crane vận chuyển xong cont trước ---> cont sau không phải chờ!
                    else:
                        Crane_Complete[k] = jobFinishTime[x] + Crane_process[k][i]
                        jobFinishTime[x] = jobFinishTime[x] + Crane_process[k][i]
                        

    AvgFinishTime = float(sum(jobFinishTime.values())/len(jobFinishTime.values()))

    ###################################################################### CMAX OF SOLUTION ############################################################

    df = pd.DataFrame([Trucksequence, Cranesequence], index=['TruckSequence', 'CraneSequence'])
    Truck_result = [0]*a
    Crane_result = [0]*c
    Truck_idx_lst = []
    Crane_idx_lst = []
    childAvgFinishTime = 0

    Genetic_dict = {}
    Reality_dict = {}
    listOfValue = []
    parent_lstcontTime = list(jobFinishTime.values())

    count = 1
    isfirst = True
    count_gene = 0

    print("\nJobs = {0}\n".format(n))
    print("Truck's mutation_rate: {0}".format(truck_rate))
    print("Crane's mutation_rate: {0}".format(crane_rate))

    # Fix this value if you want to change total generation
    while count <= 200:
        
        if isfirst:
            preChildAvgFinishTime = childAvgFinishTime

        # Mutation
        childTruckSeq, childTruckCutPosition = Truck_mutation(count, n, a)
        childCraneSeq, childCraneCutPosition = Crane_mutation(count, n, c)
        childAvgFinishTime, lst_contTime = Fitness_Caculate(childTruckSeq, childTruckCutPosition
                                     , childCraneSeq, childCraneCutPosition
                                     , truck_ready, truck_processtime
                                     , crane_ready, crane_processtime)

        
        # Mutation_rate accepted                            
        # if (mutationRate_accepted(Trucksequence, childTruckSeq, truck_rate) == False) or (mutationRate_accepted(Cranesequence, childCraneSeq, crane_rate) == False):
        #     continue

        """ This fuction use to test mutation_rate!
            If the mutation rate is not accepted:
            --> this function will show sequence's current mutation rate """   

        if (mutationRate_accepted_test(Trucksequence, childTruckSeq) < truck_rate) or (mutationRate_accepted_test(Cranesequence, childCraneSeq) < crane_rate):
            print(mutationRate_accepted_test(Trucksequence, childTruckSeq), mutationRate_accepted_test(Cranesequence, childCraneSeq))
            continue
      

        # Selective
        if (AvgFinishTime <= childAvgFinishTime) or (AvgFinishTime - childAvgFinishTime > 0.3):

            count_gene = count_gene + 1

            if childAvgFinishTime < preChildAvgFinishTime:
                isfirst = False
                preChildAvgFinishTime = childAvgFinishTime
                Reality_dict[count_gene - 1] = childAvgFinishTime

           # If there are more than 500 chromosomes --> 
            if count_gene > 1000:
                listOfValue.append(Reality_dict)
                Reality_dict = {} # Reset Dict
                isfirst = True
                pop_size = count_gene
                count_gene = 0
                Genetic_dict[count] = [AvgFinishTime]

                #ChildTruck and ChildCrane list

                Truck_result[0] = Trucksequence[1:Parent_truck_cut_position[0]+1]
                Truck_result[0].sort()
                for i in range(1, a):
                    Truck_result[i] = Trucksequence[Parent_truck_cut_position[i-1]+1:Parent_truck_cut_position[i]+1]
                    Truck_result[i].sort()

                Crane_result[0] = Cranesequence[1:Parent_crane_cut_position[0]+1]
                Crane_result[0].sort()
                for i in range(1, c):
                    Crane_result[i] = Cranesequence[Parent_crane_cut_position[i-1]+1:Parent_crane_cut_position[i]+1]
                    Crane_result[i].sort()

                df1 = pd.DataFrame([Trucksequence, Cranesequence], index=['TruckSequence', 'CraneSequence'])
                df = df.append(df1)

                print("\nGeneration:", count)
                print("Time of each container: ", list(parent_lstcontTime))
                print("Cmin = ", AvgFinishTime)
                print("Population Size of gene {0}: {1}".format(count, pop_size+1))

                count = count + 1
                continue
            else: continue

        listOfValue.append(Reality_dict)
        Reality_dict = {} # Reset Dict
        isfirst = True
        pop_size = count_gene
        count_gene = 0
        Trucksequence = childTruckSeq
        Cranesequence = childCraneSeq
        Genetic_dict[count] = [childAvgFinishTime]
        parent_lstcontTime = lst_contTime

        #ChildTruck and ChildCrane list

        Truck_result[0] = childTruckSeq[1:childTruckCutPosition[0]+1]
        Truck_result[0].sort()
        for i in range(1, a):
            Truck_result[i] = childTruckSeq[childTruckCutPosition[i-1]+1:childTruckCutPosition[i]+1]
            Truck_result[i].sort()

        Crane_result[0] = childCraneSeq[1:childCraneCutPosition[0]+1]
        Crane_result[0].sort()
        for i in range(1, c):
            Crane_result[i] = childCraneSeq[childCraneCutPosition[i-1]+1:childCraneCutPosition[i]+1]
            Crane_result[i].sort()

        df1 = pd.DataFrame([childTruckSeq, childCraneSeq], index=['TruckSequence', 'CraneSequence'])
        df = df.append(df1)

        print("\nGeneration:", count)
        print("Time of each container: ", list(lst_contTime))
        print("Cmin = ", childAvgFinishTime)
        print("Population Size of gene {0}: {1}".format(count, pop_size+1))

        AvgFinishTime = childAvgFinishTime
        count = count + 1

    for i in range(0,a):
        Truck_idx_lst.append('Truck {0}'.format(i+1))

    for i in range(0,c):
        Crane_idx_lst.append('Crane {0}'.format(i+1))


    df2 = pd.DataFrame([Truck_result[i] for i in range(0, a)], index=[Truck_idx_lst])
    df3 = pd.DataFrame([Crane_result[i] for i in range(0, c)], index=[Crane_idx_lst])
    df2 = df2.append(df3)

    # Get list_simulate for DataFrame: [jobs, what truck, what crane]

    list_of_simulate = Get_simulate_list(Trucks=Truck_result, Cranes=Crane_result, jobs=n)
    jobs_oder = [list_of_simulate[idx][0] for idx in range(0, len(list_of_simulate))]
    trucks_oder = [list_of_simulate[idx][1] for idx in range(0, len(list_of_simulate))]
    crane_oder = [list_of_simulate[idx][2] for idx in range(0, len(list_of_simulate))]

    df4 = pd.DataFrame({'Jobs': jobs_oder,
                        'What Truck?': trucks_oder,
                        'What Crane?': crane_oder})

    # Write DataFrame to Excel

    with pd.ExcelWriter(result_file) as writer:
        df.to_excel(writer, sheet_name='Sequences')
        df2.to_excel(writer, sheet_name='result')
        df4.to_excel(writer, index=False, sheet_name='Simulators')
        writer.save()

    
    print("Best Solution: ")
    print("Truck Sequence: ",childTruckSeq)
    print("Crane Sequence: ",childCraneSeq)
    print("{0} = {1}".format(" Cmin", AvgFinishTime))
    print("Population Size of gene {0}: {1}".format(count-1, pop_size+1))
    print("Total Generation: ", count-1)
    print("Result is added in 'Result.xlsx!'")
    end_time = time.time()
    print("Total time: {:0.2f} seconds".format(end_time - start_time))

    for i in range(len(listOfValue)):
        color = (rd.random(), rd.random(), rd.random())
        plt.plot(list(listOfValue[i].keys()), list(listOfValue[i].values()), linewidth='1', c=color)

    plt.title('Graph showing the relationship between Chromosomes and Fitness')
    plt.xlabel('Chromosomes')
    plt.ylabel('Fitness')
    plt.legend(loc='best')
    plt.show()

    draw_graph(Genetic_dict.keys(), Genetic_dict.values())


if __name__ == '__main__':

    # File's Directory

    path_data = 'C:\\Users\\QUOCVU\\Desktop\\VScode\\MTCNN\\data-100cont.xlsx'
    path_result = 'C:\\Users\\QUOCVU\\Desktop\\VScode\\MTCNN\\Result.xlsx'

    # Enter truck and crane number

    truck_total = int(input("Nhap so luong Truck: "))
    crane_total = int(input("Nhap so luong Crane: "))
    truck_mutation_rate = float(input("Nhap Truck's mutation rate: "))
    crane_mutation_rate = float(input("Nhap Crane's mutation rate: "))

    # Read data from data.xlsx

    data = pd.read_excel(path_data)
    jobs_list = data['Job'].tolist()
    jobs = len(jobs_list)
    truck_ready = data['YT_Ready'].tolist()
    truck_processtime = data['YT_Processing'].tolist()
    crane_ready = data['YC_Ready'].tolist()
    crane_processtime = data['YC_Processing'].tolist()

    # Run Genetic
    Genetic(jobs, truck_total, crane_total, truck_mutation_rate, 
            crane_mutation_rate,truck_ready, truck_processtime, 
            crane_ready, crane_processtime, result_file=path_result)