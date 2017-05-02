import numpy as np
import math
from PIL import Image
import imageio
import matplotlib.pyplot as plt

def create_image(board,dim,p):
    color_array = np.zeros((p*dim,p*dim,3),dtype=np.uint8)
    for i in range(dim):
        for j in range(dim):
            x = p*i
            y = p*j
            for k in range(p):
                for r in range(p):
                    if board[i,j] == 0:
                        color_array[x+k,y+r,:] = [255,255,255]
                    elif board[i,j] == 1:
                        color_array[x+k,y+r,:] = [128,128,128]
                    elif board[i,j] == 2:
                        color_array[x+k,y+r,:] = [64,64,64]
                    elif board[i,j] == 3:
                        color_array[x+k,y+r,:] = [0,0,0]
                    elif board[i,j] >= 4:
                        color_array[x+k,y+r,:] = [255,0,0]
    return color_array

def save_gif(N,name,p):
    dim = math.ceil(N**.5)+1
    board = np.zeros((dim,dim))
    center = math.ceil(dim/2)-1
    board[center,center] = N
    t = 0
    new_board = np.copy(board)
    images = [create_image(board,dim,p)]
    while (not ((new_board == board).all()) or t == 0) and t < 1000:
        t += 1
        board = np.copy(new_board)
        for i in range(1,dim-1):
            for j in range(1,dim-1):
                if board[i,j] >= 4:
                    new_board[i,j]  -= 4
                    new_board[i-1,j] += 1
                    new_board[i+1,j] += 1
                    new_board[i,j-1] += 1
                    new_board[i,j+1] += 1
        img = create_image(board,dim,p)
        images += [img]

    imageio.mimsave('gifs/' + name+'.gif',images)

def save_image(N,name,p,max_t=1000000):
    dim = math.ceil(N**.5)+1
    board = np.zeros((dim,dim))
    center = math.ceil(dim/2) - 1
    board[center,center] = N
    new_board = np.copy(board)
    t = 0
    while (not ((new_board == board).all()) or t == 0) and t <= max_t:
        t += 1
        board = np.copy(new_board)
        for i in range(1,dim-1):
            for j in range(1,dim-1):
                if board[i,j] >= 4:
                    new_board[i,j]  -= 4
                    new_board[i-1,j] += 1
                    new_board[i+1,j] += 1
                    new_board[i,j-1] += 1
                    new_board[i,j+1] += 1
    img =  Image.fromarray(create_image(new_board,dim,p),'RGB')
    img.save('pics/'+ name + '.png')

def empty(board,t):
    return None

def play_game(N,func = empty):
    dim = math.ceil(N**.5)+1
    board = np.zeros((dim,dim))
    center = math.ceil(dim/2) - 1
    board[center,center] = N
    new_board = np.copy(board)
    t = 0
    info = [func(board,t)]
    while (not ((new_board == board).all()) or t == 0) and t < 10000:
        t += 1
        board = np.copy(new_board)
        for i in range(1,dim-1):
            for j in range(1,dim-1):
                if board[i,j] >= 4:
                    new_board[i,j]  -= 4
                    new_board[i-1,j] += 1
                    new_board[i+1,j] += 1
                    new_board[i,j-1] += 1
                    new_board[i,j+1] += 1
        info += [func(board,t)]
    return(new_board,t,info)

def get_radius_function(low,high,func = empty):
    ts = []
    diam = []
    for n in range(low,high+1):
        board,t,info = play_game(n)
        ts += [(t,n)]
        dim = math.ceil(n**.5)+1
        center = math.ceil(dim/2) - 1
        look_at = board[center,:]
        width = len(look_at)
        edge = 0
        for i in look_at:
            if i == 0:
                edge += 1
            else:
                break
        for j in look_at[::-1]:
            if j == 0:
                edge += 1
            else:
                break
        diam += [(width-edge,n)]
    return(diam,ts)

def get_steps(diam,up=True):
    steps = []
    for i in range(1,len(diam)):
        if diam[i][0] > diam[i-1][0]:
            if up:
                steps += [diam[i]]
            else:
                steps += [diam[i-1]]
    return(steps)

def make_a_lot_of_images(chips,high):
    for i in range(1,high):
        save_image(chips,str(high)+'chips_at_time_step_'+str(i),10,i)

def get_red_radius(board,t):
    distances = []
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if board[i,j] >= 4:
                distances += [[board[i,j],math.fabs(i) + math.fabs(j),t,i,j]]
    distances = sorted(distances, key = lambda x: -x[1])
    if not distances:
        return None
    return distances[0]
diam, ts = get_radius_function(1000,2000)
steps_up = get_steps(diam,True)
# steps_down = get_steps(diam,False)
#board, t, results = play_game(1000,get_red_radius)
y = []
x = []
for i in range(1,len(steps_up)):
    if steps_up[i] != None:
        y += [steps_up[i][1]-steps_up[i-1][1]]
        x += [steps_up[i][0]]
plt.plot(x,y,'k')
plt.show()
