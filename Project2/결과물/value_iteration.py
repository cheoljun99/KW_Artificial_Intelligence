
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

def value_iteration(grid_width, grid_height, action, iteration_num, reward=-1, dis=1):
    
    # 테이블 초기화 진행
    post_value_table = np.zeros([grid_height, grid_width], dtype=float)
    
    if iteration_num == 0:#예외처리
        print('Iteration: {} \n{}\n'.format(iteration_num, post_value_table))
        return post_value_table

        
    # iteration 실행  
    for iteration in range(iteration_num+1):
        if iteration == 0:
            print('Iteration: {} \n{}\n'.format(iteration, post_value_table))
        else:
            next_value_table = np.zeros([grid_height, grid_width], dtype=float)
            for i in range(grid_height):
                for j in range(grid_width):
                    if i == j and ((i == 6)):
                        value_t_list= []
                        value_t_list.append(0)#도착점
                    else :
                        value_t_list= []
                        for act in action:
                            i_, j_ = get_post_state([i,j], act)
                            if((i_==0 and j_==2)or(i_==1 and j_==2)or(i_==3 and j_==4)or(i_==3 and j_==5)or(i_==6 and j_==2)or(i_==6 and j_==3)):#함정에 빠진경우
                                #print("ok")
                                value = (-100 + dis*post_value_table[i_][j_])
                                value_t_list.append(value)
                            else:
                                value = ((reward) + dis*post_value_table[i_][j_])
                                value_t_list.append(value)
                        
                    next_value_table[i][j] = max(value_t_list)

            # 결과 출력
            print('Iteration: {} \n{}\n'.format(iteration, next_value_table))
            post_value_table = next_value_table
    return 


grid_width = 7 #너비
grid_height = 7 # 높이
action = [0, 1, 2, 3] # up, down, left, right
value_iteration(grid_width, grid_height, action, 20)
