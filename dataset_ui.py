import os
import random as rd
import pygame as pg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


pg.init()


L = 28
WIN = pg.display.set_mode((L*10,L*10))
pg.display.set_caption("Numbers")
print(rd.randint(0, 9))

def draw_gameboard(board):
    for y in range(L):
        for x in range(L):
            n = board[y][x]
            pg.draw.rect(WIN, (int(255*n),int(255*n),int(255*n)), (x*10, y*10, 10, 10))


def save(export, board):
    '''Save image:
    Save the pygame window as an image in the data folder'''
    WIN = pg.display.set_mode((L,L))
    for y in range(L):
        for x in range(L):
            n = board[y][x]
            pg.draw.rect(WIN, (int(255*n),int(255*n),int(255*n)), (x, y, 1, 1))    


    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists("data/{}".format(export)):
        os.mkdir("data/{}".format(export))
    for i in range(100):
        if not os.path.exists("data/{}/{}.png".format(export, i)):
            pg.image.save(WIN, "data/{}/{}.png".format(export, i))
            break


def create_df():
    '''Dataframe creation:
    Create a dataframe with the data and the label of each image'''
    df = pd.DataFrame(columns=['data', 'label'])
    for i in range(10):
        folder = str(i)
        for file in os.listdir('data/' + folder):
            img = plt.imread('data/' + folder + '/' + file)
            gray = (np.mean(img, axis=2).reshape(1,-1)*255).astype(np.uint8).tolist()[0]
            df.loc[len(df)] = [gray, folder]
    return df


def save_image(path,n):
    '''Save image:
    Save an image in a csv file'''
    img=plt.imread(path)
    gray=(np.mean(img,axis=2).reshape(1,-1)*255).astype(np.uint8).tolist()[0]
    df=pd.DataFrame(columns=['data','label'])
    df.loc[0]=[gray,n]
    df.to_csv(f"{n}.csv",index=False)


def main():
    '''Main function:
    Create a window to draw numbers and save them in the data folder'''
    board = [[0 for x in range(L)] for y in range(L)]
    clock = pg.time.Clock()
    export = -1
    run = True
    while run:
        clock.tick(120)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False
            if event.type == pg.KEYUP:
                if event.key == pg.K_0:
                    export = 0
                if event.key == pg.K_1:
                    export = 1
                if event.key == pg.K_2:
                    export = 2
                if event.key == pg.K_3:
                    export = 3
                if event.key == pg.K_4:
                    export = 4
                if event.key == pg.K_5:
                    export = 5
                if event.key == pg.K_6:
                    export = 6
                if event.key == pg.K_7:
                    export = 7
                if event.key == pg.K_8:
                    export = 8
                if event.key == pg.K_9:
                    export = 9
        mouse = pg.mouse.get_pressed()
        if mouse[0]:
            x, y = pg.mouse.get_pos()
            x , y = x//10, y//10
            board[y][x] = 1
            if x < L-1:
                board[y][x+1] = (1 + board[y][x+1])/2
            if x > 0:
                board[y][x-1] = (1 + board[y][x-1])/2
            if y < L-1:
                board[y+1][x] = (1 + board[y+1][x])/2
            if y > 0:
                board[y-1][x] = (1 + board[y-1][x])/2
        if mouse[2]:
            x, y = pg.mouse.get_pos()
            x , y = x//10, y//10
            board[y][x] = 0
            if x < L-1:
                board[y][x+1] = 0
            if x > 0:
                board[y][x-1] = 0
            if y < L-1:
                board[y+1][x] = 0
            if y > 0:
                board[y-1][x] = 0
        
        if export != -1:
            save(export, board)
            WIN = pg.display.set_mode((L*10,L*10))
            board = [[0 for x in range(L)] for y in range(L)]
            export = -1
        
        draw_gameboard(board)
        pg.display.update()

    pg.quit()


if __name__ == "__main__":
    main()
    df = create_df()
    df.to_csv("data/data.csv", index=False)
