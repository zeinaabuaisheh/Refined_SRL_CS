

import pygame

pygame.init()  # Pygame initialialization

# region COLOR DEFINITION
white = (255, 255, 255)
black = (0, 0, 0)
grey = (80, 80, 80)
red = (255, 0, 0)
blue = (0, 0, 255)
green = (0, 255, 0)
yellow = (250, 250, 0)
pink = (250, 105, 180)
# endregion

# region TEXT FONTS DEFINITION
smallfont = pygame.font.SysFont('comicsansms', 13)
smallfont_act = pygame.font.SysFont('arial', 13)
mediumfont_act = pygame.font.SysFont('arial', 18, bold=True)
pygame.font.init()
# endregion

# region DISPLAY FUNCTIONS
def show_Alg(alg, screen):
    text = smallfont.render("Alg: " + alg, True, black)
    screen.blit(text, [5 + 90 * 0, 0])

def show_Samples(sample, screen):
    text = smallfont.render("Sample: " + str(sample), True, black)
    screen.blit(text, [60+100*1, 0])

def show_Level(level, screen):
    text = smallfont.render("Episode: " + str(level), True, black)
    screen.blit(text, [50+100*2, 0])

def show_Score(score, screen):
    text = smallfont.render("Score: " + str(score), True, black)
    screen.blit(text, [50+100*3, 0])

def show_Steps(steps, screen):
    text = smallfont.render("Steps: " + str(steps), True, black)
    screen.blit(text, [50+100*4, 0])

def show_Percent(percent, screen):
    text = smallfont.render("Percent: " + str(['%.2f' % elem for elem in percent]), True, black)
    screen.blit(text, [5, 30 * 4])

def show_Steps_list(steps_list, screen):
    text = smallfont.render("Steps_list: " + str(steps_list), True, black)
    screen.blit(text, [5, 30 * 1])

def show_Act_List(act_list, screen):
    text = smallfont_act.render("act_list: " + str(act_list), True, black)
    screen.blit(text, [5, 30 * 2])

def show_Action(act, screen):
    text = smallfont_act.render("Chosen Action: " + act, True, black)
    screen.blit(text, [5, 30 * 3])

def show_Env(env, screen):
    text = mediumfont_act.render("Environment:  " + str(env), True, black)
    screen.blit(text, [50, 30 * 5])
# endregion
