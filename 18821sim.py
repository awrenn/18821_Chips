import numpy as np
import math
from PIL import Image
import imageio
import matplotlib.pyplot as plt


def create_image(board,dim,p,circ = False):
    color_array = np.zeros((p*dim,p*dim,3),dtype=np.uint8)
    center = math.ceil(dim/2)-1
    furthest = 0
    for i in range(dim):
        for j in range(dim):
            dist = math.ceil(((i-center)**2 + (j-center)**2)**.5)
            if dist > furthest and board[i,j] > 0:
                furthest = dist
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
    if circ:
        print(furthest)
        a_half = furthest
        p_center = center*p
        for i in range(dim):
            for j in range(dim):
                x = p*i
                y = p*j
                half = a_half*p
                rad = ((i-center)**2+(j-center)**2)**.5
                goal = furthest
                if math.ceil(rad) == math.ceil(goal):
                    for k in range(p):
                        for r in range(p):
                            if board[i,j] > 0:
                                color_array[x+k,y+r,:] = [255,0,0]
                            else:
                                color_array[x+k,y+r,:] = [0,255,0]
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

def save_image(N,name,p,max_t=1000000,circ = False):
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
    img =  Image.fromarray(create_image(new_board,dim,p,circ),'RGB')
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
    while (not ((new_board == board).all()) or t == 0) and t < 100000:
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

def get_occupied_length(line):
    edge = 0
    for i in line:
        if i == 0:
            edge += 1
        else:
            break
    for j in line[::-1]:
        if j == 0:
            edge += 1
        else:
            break
    return(len(line)-edge)

def get_chip_count(line):
    chips = 0
    for i in line:
        chips += i
    return chips

def get_line_densities(board):
    dims = board.shape
    densities = []
    for line in range(board.shape[0]):
        look_at = board[line,:]
        length = get_occupied_length(look_at)
        chips = get_chip_count(look_at)
        densities += [chips/length]
    return densities

def get_box_densities(board):
    dims = board.shape
    densities = []
    center = math.ceil(dims[0]/2)-1
    for i in range(0,math.floor(dims[0]/2)):
        look_at = board[center-i:center+i+1,center-i:center+i+1]
        chips = 0
        for x in range(look_at.shape[0]):
            for y in range(look_at.shape[1]):
                chips += look_at[x,y]
        area = (2*i+1)**2
        densities += [chips/area]
    return densities

def get_circ_den(board,N):
    dim = math.ceil(N**.5)+1
    center = math.ceil(dim/2)-1
    furthest = 0
    for i in range(dim):
        for j in range(dim):
            dist = math.ceil(((i-center)**2+(j-center)**2)**.5)
            if dist > furthest and board[i,j] > 0:
                furthest = dist
    chips = 0
    area = 0
    a_half = furthest
    for i in range(dim):
        for j in range(dim):
            rad = ((i-center)**2+(j-center)**2)**.5
            goal = furthest
            if math.ceil(rad) <= math.ceil(goal):
                area += 1
                chips += board[i,j]
    return chips/area

def get_circ_den_list(board,N):
    dim = math.ceil(N**.5)+1
    center = math.ceil(dim/2)-1
    furthest = 0
    for i in range(dim):
        for j in range(dim):
            dist = math.floor(((i-center)**2+(j-center)**2)**.5)
            if dist > furthest and board[i,j] > 0:
                furthest = dist
    x = []
    y = []
    for r in range(1,furthest):
        area = 0
        chips = 0
        for i in range(dim):
            for j in range(dim):
                rad = ((i-center)**2+(j-center)**2)**.5
                goal = r
                if math.floor(rad) <= math.floor(goal):
                    area += 1
                    chips += board[i,j]
        x += [r]
        y += [chips/area]
    return x,y




def save_gif_exp(N,name,p):
    dim = math.ceil(N**.7)+1
    board = np.zeros((dim,dim))
    center = math.ceil(dim/2)-1
    l_center = math.ceil(3*dim/7)-1
    r_center = math.ceil(4*dim/7)-1
    board[center,l_center] = N/2
    board[center,r_center] = N/2
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

#save_image(10000,"10kborder",20,circ=True)


 N = 1000000


# x = []
# y = []
# diams, ts = get_radius_function(1,1000)
# for t in ts:
#     x += [t[0]]
#     y += [t[1]]
#
# plt.plot(x,y)
# plt.show()
