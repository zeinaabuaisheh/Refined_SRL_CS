
import numpy as np
import random
import pygame

# region ENVIRONMENT CONFIGURATION
def environment_conf(s_env):
    if s_env == 1:
        v_max = 4
        h_max = 5
        x_agent = 1
        y_agent = 2
        m_nega = np.matrix([[0, 0, 0],
                            [0, 1, 0]])
        m_posi = np.matrix([[0, 1, 0],
                            [0, 0, 0]])

    elif s_env == 2:
        v_max = 4
        h_max = 5
        x_agent = 1
        y_agent = 2
        m_nega = np.matrix([[0, 0, 0],
                            [0, 0, 1]])
        m_posi = np.matrix([[0, 0, 1],
                            [0, 0, 0]])

    elif s_env == 3:
        v_max = 4
        h_max = 5
        x_agent = 1
        y_agent = 2
        m_nega = np.matrix([[1, 0, 0],
                            [0, 0, 0]])
        m_posi = np.matrix([[0, 1, 0],
                            [0, 0, 0]])

    elif s_env == 4:
        v_max = 4
        h_max = 4
        x_agent = 1
        y_agent = 1
        m_nega = np.matrix([[0, 0],
                            [0, 0]])
        m_posi = np.matrix([[0, 0],
                            [0, 1]])

    elif s_env == 5:
        v_max = 5
        h_max = 5
        x_agent = 2
        y_agent = 2
        m_nega = np.zeros(shape=(v_max - 2, h_max - 2))
        m_posi = np.zeros(shape=(v_max - 2, h_max - 2))
        while (True):
            x = random.randrange(0, h_max - 2)
            y = random.randrange(0, v_max - 2)
            if x != x_agent-1 or y != y_agent-1:
                element = (x, y)
                break
        m_posi[element] = 1

    elif s_env == 6:
        v_max = 7
        h_max = 7
        x_agent = 3
        y_agent = 3
        m_nega = np.zeros(shape=(v_max - 2, h_max - 2))
        m_posi = np.zeros(shape=(v_max - 2, h_max - 2))
        while (True):
            x = random.randrange(0, h_max - 2)
            y = random.randrange(0, v_max - 2)
            if x != x_agent - 1 or y != y_agent - 1:
                element = (x, y)
                break
        m_posi[element] = 1

    elif s_env == 7:
        v_max = 9
        h_max = 9
        x_agent = 4
        y_agent = 4
        m_nega = np.zeros(shape=(v_max - 2, h_max - 2))
        m_posi = np.zeros(shape=(v_max - 2, h_max - 2))
        while (True):
            x = random.randrange(0, h_max - 2)
            y = random.randrange(0, v_max - 2)
            if x != x_agent - 1 or y != y_agent - 1:
                element = (x, y)
                break
        m_posi[element] = 1

    elif s_env == 8:
        v_max = 5
        h_max = 5
        x_agent = 2
        y_agent = 2
        m_nega = np.matrix([[0, 0, 0],
                            [0, 0, 0],
                            [1, 0, 1]])
        m_posi = np.matrix([[1, 0, 1],
                            [0, 0, 0],
                            [0, 0, 0]])

    elif s_env == 9:
        v_max = 5
        h_max = 5
        x_agent = 2
        y_agent = 2
        m_nega = np.matrix([[1, 0, 0],
                            [0, 0, 0],
                            [0, 0, 1]])
        m_posi = np.matrix([[0, 0, 1],
                            [0, 0, 0],
                            [1, 0, 0]])

    elif s_env == 10:
        v_max = 9
        h_max = 9
        x_agent = 4
        y_agent = 4
        m_nega = np.matrix([[1, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 1]])
        m_posi = np.matrix([[0, 0, 1, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 1, 0, 0]])

    elif s_env == 11:
        v_max = 9
        h_max = 9
        x_agent = 4
        y_agent = 4
        element_list = []
        for n in range(14):
            while(True):
                x = random.randrange(0,7)
                y = random.randrange(0,7)
                if x != 3 and y != 3 and (x,y) not in element_list:
                    element = (x, y)
                    break
            element_list.append(element)

        m_nega = np.zeros(shape=(v_max-2, h_max-2))
        m_posi = np.zeros(shape=(v_max-2, h_max-2))
        half = len(element_list) / 2
        nega_list = element_list[:int(half)]
        posi_list = element_list[int(half):]
        for ele in nega_list:
            m_nega[ele] = 1
        for ele in posi_list:
            m_posi[ele] = 1

    elif s_env == 12:
        v_max = 3
        h_max = 5
        x_agent = 2
        y_agent = 1
        m_nega = np.matrix([1, 0, 0])
        m_posi = np.matrix([0, 0, 1])

    elif s_env == 13:
        v_max = 3
        h_max = 5
        x_agent = 2
        y_agent = 1
        m_nega = np.matrix([0, 0, 0])
        m_posi = np.matrix([1, 0, 1])

    elif s_env == 14:
        v_max = 3
        h_max = 6
        x_agent = 2
        y_agent = 1
        m_nega = np.matrix([1, 0, 0, 0])
        m_posi = np.matrix([0, 0, 0, 1])

    elif s_env == 15:
        v_max = 3
        h_max = 6
        x_agent = 2
        y_agent = 1
        m_nega = np.matrix([0, 0, 0, 0])
        m_posi = np.matrix([1, 0, 0, 1])

    elif s_env == 16:
        v_max = 3
        h_max = 7
        x_agent = 3
        y_agent = 1
        m_nega = np.matrix([1, 0, 0, 0, 0])
        m_posi = np.matrix([0, 0, 0, 0, 1])

    elif s_env == 17:
        v_max = 3
        h_max = 7
        x_agent = 3
        y_agent = 1
        m_nega = np.matrix([0, 0, 0, 0, 0])
        m_posi = np.matrix([1, 0, 0, 0, 1])

    elif s_env == 18:
        v_max = 3
        h_max = 9
        x_agent = 4
        y_agent = 1
        m_nega = np.matrix([1, 0, 0, 0, 0, 0, 0])
        m_posi = np.matrix([0, 0, 0, 0, 0, 0, 1])

    elif s_env == 19:
        v_max = 3
        h_max = 9
        x_agent = 4
        y_agent = 1
        m_nega = np.matrix([0, 0, 0, 0, 0, 0, 0])
        m_posi = np.matrix([1, 0, 0, 0, 0, 0, 1])

    elif s_env == 20:
        v_max = 5
        h_max = 5
        x_agent = 2
        y_agent = 2
        m_nega = np.matrix([[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]])
        m_posi = np.matrix([[1, 0, 1],
                            [0, 0, 0],
                            [0, 1, 0]])

    elif s_env == 21:
        v_max = 5
        h_max = 5
        x_agent = 2
        y_agent = 2
        m_nega = np.matrix([[0, 1, 0],
                            [0, 0, 0],
                            [1, 0, 1]])
        m_posi = np.matrix([[1, 0, 1],
                            [0, 0, 0],
                            [0, 1, 0]])

    elif s_env == 22:
        v_max = 5
        h_max = 5
        x_agent = 2
        y_agent = 2
        m_nega = np.matrix([[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]])
        m_posi = np.matrix([[1, 0, 1],
                            [0, 0, 0],
                            [1, 0, 1]])

    if s_env == 31:
        v_max = 5
        h_max = 5
        x_agent = 1
        y_agent = 2
        m_nega = np.matrix([[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]])
        m_posi = np.matrix([[0, 1, 0],
                            [0, 0, 0],
                            [0, 0, 0]])

    elif s_env == 32:
        v_max = 5
        h_max = 5
        x_agent = 1
        y_agent = 2
        m_nega = np.matrix([[0, 0, 0],
                            [0, 0, 1],
                            [0, 0, 0]])
        m_posi = np.matrix([[0, 0, 1],
                            [0, 0, 0],
                            [0, 0, 0]])

    elif s_env == 33:
        v_max = 5
        h_max = 5
        x_agent = 1
        y_agent = 2
        m_nega = np.matrix([[1, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]])
        m_posi = np.matrix([[0, 1, 0],
                            [0, 0, 0],
                            [0, 0, 0]])

    else:
        pass

    "INSTANCE THE wall_list"
    wall_list = []
    for y in range(v_max):
        for x in range(h_max):
            if y == v_max - 1 or y == 0 or x == h_max - 1 or x == 0:
                wall = Wall('./images/wall', x, y)
                wall_list.append(wall)
    "INSTANCE THE AGENT"
    agent = Agent('./images/agent', x_agent, y_agent)

    "INSTANCE POSITIVE OBJECTS"
    positivo_list = []
    for x in range(m_posi.shape[0]):
        for y in range(m_posi.shape[1]):
            if m_posi[x, y] == 1:
                positivo = Positivo('./images/positivo', y + 1, x + 1)
                positivo_list.append(positivo)

    "INSTANCE NEGATIVE OBJECTS"
    negativo_list = []
    for x in range(m_nega.shape[0]):
        for y in range(m_nega.shape[1]):
            if m_nega[x, y] == 1:
                negativo = Negativo('./images/negativo', y + 1, x + 1)
                negativo_list.append(negativo)

    return negativo_list, positivo_list, agent, wall_list, h_max, v_max
# endregion


# region DRAW OBJECTS
x_zero_screen = 50
y_zero_screen = 180
size_obj = 37
def draw_objects(agent, positivo_list, negativo_list, wall_list, screen):
    # Class.Grid.draw_grid(screen) # Uncomment to display a Grid
    for i in positivo_list:  # POSITIVO
        screen.blit(i.icon, (i.pos[0] * size_obj + x_zero_screen, y_zero_screen + i.pos[1] * size_obj))
    for i in negativo_list:  # NEGATIVO
        screen.blit(i.icon, (i.pos[0] * size_obj + x_zero_screen, y_zero_screen + i.pos[1] * size_obj))
    screen.blit(agent.icon, (agent.pos[0] * size_obj + x_zero_screen, y_zero_screen + agent.pos[1] * size_obj))  # AGENT
    for i in wall_list:  # WALL
        screen.blit(i.icon, (i.pos[0] * size_obj + x_zero_screen, y_zero_screen + i.pos[1] * size_obj))
# endregion

# region CREATE THE STATE FROM THE ENVIRONMENT
def update_state(h_max, v_max, agent, positivo_list, negativo_list, wall_list):
    state = np.zeros((v_max, h_max)).astype(np.int16)
    for i in positivo_list:
        state[i.pos[1]][i.pos[0]] = 60  # SYMBOL 60 POSITIVE
    for i in negativo_list:
        state[i.pos[1]][i.pos[0]] = 180  # SYMBOL 180 NEGATIVE
    for i in wall_list:
        state[i.pos[1]][i.pos[0]] = 255  # SYMBOL 255
    # state[agent.pos[1]][agent.pos[0]] = 120  # SYMBOL 60
    return state
    # TODO I have to check if this v_max and h_max have to be declared everytime
# endregion



white = (255,255,255)
black = (0,0,0)
grey = (80,80,80)
red = (255,0,0)
dark_red = (155,0,0)
blue = (0,0,255)
green = (0,255,0)
yellow = (250,250,0)
pink = (250,105,180)

'''Size of the game'''
size = 'Big'
if size == 'Big':
    m = 5 # Big
    x_g,y_g = 30, 30 # Big
    w,h = 30, 30 # Big
else:
    m = 1 # Small
    x_g,y_g = 5, 5 # Small
    w,h = 5, 5 # Small

class Grid(object):
    grid_w = w
    w = 6
    grid_h = h
    h = 5

    def draw_grid(screen):
        pygame.draw.rect(screen, black, [x_g, y_g, Grid.grid_w , Grid.grid_h])
        for row in range(Grid.h):
            for column in range(Grid.w):
                color = white
                pygame.draw.rect(screen,
                                 color,
                                 [x_g + (m + Grid.grid_w) * column + m,
                                  y_g + (m + Grid.grid_h) * row + m,
                                  Grid.grid_w, Grid.grid_h])

class Wall(pygame.Rect):
    color = grey
    def __init__(self,img,x,y):
        self.icon = pygame.image.load(str(img) + size + '.png')
        self.pos = [x, y]
        self.x = x
        self.y = y

class Obj:
    def __init__(self, tp, loc):
        self.tp = tp
        self.loc = loc

class Agent(pygame.Rect):
    color = yellow
    def __init__(self,img,x,y):
        self.icon = pygame.image.load(str(img) + size + '.png')
        self.pos = [x, y]
        self.x = x
        self.y = y

    def try_move(self, dir, wall_list):
        x_step = 0
        y_step = 0
        past_pos = self.pos
        if dir == 'up':
            y_step = - 1
        if dir == 'down':
            y_step = + 1
        if dir == 'left':
            x_step = - 1
        if dir == 'right':
            x_step = + 1
        fut_x = self.pos[0] + x_step
        fut_y = self.pos[1] + y_step
        fut_pos = [fut_x, fut_y]

        for wall in wall_list:
            if fut_pos == wall.pos:
                self.pos = past_pos
                break
            else:
                self.pos = fut_pos

class Start(pygame.Rect):
    color = blue
    def __init__(self,img,x,y):
         self.icon = pygame.image.load(str(img) + size +  '.png')
         self.pos = [x, y]

class Negativo(pygame.Rect):
    color = pink
    def __init__(self,img,x,y):
         self.icon = pygame.image.load(str(img) + size + '.png')
         self.pos = [x, y]
         self.x = x
         self.y = y

class Positivo(pygame.Rect):
    color = green
    def __init__(self,img,x,y):
         self.icon = pygame.image.load(str(img) + size + '.png')
         self.pos = [x, y]
         self.x = x
         self.y = y
