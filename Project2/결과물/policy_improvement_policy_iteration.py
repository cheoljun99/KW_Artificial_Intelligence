import numpy as np

def get_post_state(state, action):
    
    #구하고자하는 value의 위치에서 동서남북으로 이전 값들의 위치를 가져온다
    action_grid = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    state[0]+=action_grid[action][0]
    state[1]+=action_grid[action][1]
    
    #영역 밖았으로 나간 경우를 체크
    if state[0] < 0 :
        state[0] = 0
    elif state[0] > 6 :
        state[0] = 6
    
    if state[1] < 0 :
        state[1] = 0
    elif state[1] > 6 :
        state[1] = 6
    
    return state[0], state[1]

def print_arrow(action, policy, grid_width):

    #화살표를 그리는 함수
    grid_height = grid_width
    action_match = ['↑   ', '↓   ', '←   ', '→   ', '↑↓  ', '↑←  ', '↑→  ', '↓←  ', '↓→  ','←→  ','←↑→ ','↑↓→ ','←↑↓ ','←↓→ ','←↑↓→']
    action_table = []
    for i in range(grid_height):
           for j in range(grid_width):
                if i==j and ((i==6)):
                    action_table.append('T    ')
                else:
                    #policy에 따른 할 수 있는 action 찾기

                    up_count=0
                    down_count=0
                    left_count=0
                    right_count=0

                    for count in range(len(action)):
                        if(policy[i][j][count]>0 and count==0):
                            up_count=up_count+1
                        elif(policy[i][j][count]>0 and count==1):
                            down_count=down_count+1
                        elif(policy[i][j][count]>0 and count==2):
                            left_count=left_count+1
                        elif(policy[i][j][count]>0 and count==3):
                            right_count=right_count+1  

                    if(up_count==1 and down_count==0 and left_count==0 and right_count==0):#↑
                        action_table.append(action_match[0])
                    if(up_count==0 and down_count==1 and left_count==0 and right_count==0):#↓
                        action_table.append(action_match[1])
                    if(up_count==0 and down_count==0 and left_count==1 and right_count==0):#←
                        action_table.append(action_match[2])
                    if(up_count==0 and down_count==0 and left_count==0 and right_count==1):#→
                        action_table.append(action_match[3])

                    if(up_count==1 and down_count==1 and left_count==0 and right_count==0):#↑↓
                        action_table.append(action_match[4])
                    if(up_count==1 and down_count==0 and left_count==1 and right_count==0):#↑←
                        action_table.append(action_match[5])    
                    if(up_count==1 and down_count==0 and left_count==0 and right_count==1):#↑→
                        action_table.append(action_match[6])
                    if(up_count==0 and down_count==1 and left_count==1 and right_count==0):#↓←
                        action_table.append(action_match[7])
                    if(up_count==0 and down_count==1 and left_count==0 and right_count==1):#↓→
                        action_table.append(action_match[8])
                    if(up_count==0 and down_count==0 and left_count==1 and right_count==1):#←→
                        action_table.append(action_match[9])   

                    if(up_count==1 and down_count==0 and left_count==1 and right_count==1):#←↑→
                        action_table.append(action_match[10])
                    if(up_count==1 and down_count==1 and left_count==0 and right_count==1):#↑↓→
                        action_table.append(action_match[11])
                    if(up_count==1 and down_count==1 and left_count==1 and right_count==0):#←↑↓
                        action_table.append(action_match[12])
                    if(up_count==0 and down_count==1 and left_count==1 and right_count==1):#←↓→
                        action_table.append(action_match[13])

                    if(up_count==1 and down_count==1 and left_count==1 and right_count==1):#←↑↓→
                        action_table.append(action_match[14])

    action_table=np.asarray(action_table).reshape((grid_height, grid_width))   
    print('at each state, chosen action is :\n{}\n'.format(action_table))

def policy_improvement_policy_iteration(value, action, policy, grid_width,iteration_num):
    
    grid_height = grid_width

    if iteration_num == 0:
        print_arrow(action, policy, grid_width)
        return 
    # Q-functiom을 사용하여 policy 재정의
    for i in range(grid_height):
        for j in range(grid_width):
            q_func_list=[]
            if i==j and ((i==6)):
                policy[i][j] = [0] * len(action)
            else:
                for k in range(len(action)):
                    i_, j_ = get_post_state([i, j], k)
                    q_func_list.append(value[i_][j_])
                max_actions = [action_v for action_v, x in enumerate(q_func_list) if x == max(q_func_list)] 

                # update policy
                policy[i][j]= [0]*len(action) # initialize q-func_list
                for y in max_actions :
                    policy[i][j][y] = (1 / len(max_actions))
    
    # 재정의 된 policy를 이용하여 action을 그래프로 출력                      
    print_arrow(action, policy, grid_width)
    return

    
def policy_evaluation(grid_width, grid_height, action, policy, iteration_num, reward=-1, dis=1):
    
   # 테이블 초기화 진행
    post_value_table = np.zeros([grid_height, grid_width], dtype=float)
    
    
    if iteration_num == 0:#예외처리
        print('Iteration: {} \n{}\n'.format(iteration_num, post_value_table))
        policy_improvement_policy_iteration(post_value_table, action, policy,grid_width,iteration_num)
        return 

    # iteration 실행   
    for iteration in range(iteration_num+1):
        if iteration == 0:
            print('Iteration: {} \n{}\n'.format(iteration, post_value_table))
            policy_improvement_policy_iteration(post_value_table, action, policy,grid_width,iteration_num)
        else:
            next_value_table = np.zeros([grid_height, grid_width], dtype=float)
            for i in range(grid_height):
                for j in range(grid_width):
                    if i == j and ((i == 6)):#도착점
                        value_t = 0
                    else :
                        value_t = 0
                        for act in action:
                            i_, j_ = get_post_state([i,j], act)
                            if((i_==0 and j_==2)or(i_==1 and j_==2)or(i_==3 and j_==4)or(i_==3 and j_==5)or(i_==6 and j_==2)or(i_==6 and j_==3)):#함정에 빠진경우
                                #print("ok")
                                value = policy[i][j][act] * (-100 + dis*post_value_table[i_][j_])
                                value_t += value
                            else:
                                value = policy[i][j][act] * (reward + dis*post_value_table[i_][j_])
                                value_t += value
                    next_value_table[i][j] = round(value_t,3)

            # 결과 출력
            print('Iteration: {} using Greedy Policy\n{}\n'.format(iteration, next_value_table ))
            policy_improvement_policy_iteration(next_value_table, action, policy,grid_width,iteration_num)
            post_value_table = next_value_table
    return

grid_width = 7 #너비
grid_height = 7 # 높이
action = [0, 1, 2, 3] # ↑, ↓, ←, →
policy = np.empty([grid_height, grid_width, len(action)], dtype=float)
for i in range(grid_height):# 초기 policy 설정
    for j in range(grid_width):
        for k in range(len(action)):
            if i==j and ( (i==6)):
                policy[i][j]=0.00
            else :
                policy[i][j]=0.25
policy[6][6] = [0] * len(action)

policy_evaluation(grid_width, grid_height, action, policy,16)# make policy improvement environment and execute policy improvement