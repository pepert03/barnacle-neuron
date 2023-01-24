import os
import pygame as pg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


pg.init()
RESOLUTION = 28
INCREASE = 10
screen = pg.display.set_mode((RESOLUTION*INCREASE,RESOLUTION*INCREASE))
pg.display.set_caption("Dataset UI")

def draw_grid(board):
    for y in range(RESOLUTION):
        for x in range(RESOLUTION):
            n = board[y][x]
            pg.draw.rect(screen, (int(255*n),int(255*n),int(255*n)), (x*INCREASE, y*INCREASE, INCREASE, INCREASE))


def save(export, board):
    '''Save image:
    Save the pygame screendow as an image in the data folder'''
    screen = pg.display.set_mode((RESOLUTION,RESOLUTION))
    for y in range(RESOLUTION):
        for x in range(RESOLUTION):
            n = board[y][x]
            pg.draw.rect(screen, (int(255*n),int(255*n),int(255*n)), (x, y, 1, 1))    


    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists("data/{}".format(export)):
        os.mkdir("data/{}".format(export))
    for i in range(200):
        if not os.path.exists("data/{}/{}.png".format(export, i)):
            pg.image.save(screen, "data/{}/{}.png".format(export, i))
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
    Create a screendow to draw numbers and save them in the data folder'''
    board = [[0 for x in range(RESOLUTION)] for y in range(RESOLUTION)]
    clock = pg.time.Clock()
    export = -1
    run = True
    while run:
        clock.tick(120)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False
            if event.type == pg.KEYUP:
                for i in range(10):
                    if event.key == pg.K_0 + i:
                        export = i
        mouse = pg.mouse.get_pressed()
        if mouse[0]:
            x, y = pg.mouse.get_pos()
            x , y = x//INCREASE, y//INCREASE
            board[y][x] = 1
            if x < RESOLUTION-1:
                board[y][x+1] = (1 + board[y][x+1])/2
            if x > 0:
                board[y][x-1] = (1 + board[y][x-1])/2
            if y < RESOLUTION-1:
                board[y+1][x] = (1 + board[y+1][x])/2
            if y > 0:
                board[y-1][x] = (1 + board[y-1][x])/2
        if mouse[2]:
            x, y = pg.mouse.get_pos()
            x , y = x//INCREASE, y//INCREASE
            board[y][x] = 0
            if x < RESOLUTION-1:
                board[y][x+1] = 0
            if x > 0:
                board[y][x-1] = 0
            if y < RESOLUTION-1:
                board[y+1][x] = 0
            if y > 0:
                board[y-1][x] = 0
        
        if export != -1:
            save(export, board)
            screen = pg.display.set_mode((RESOLUTION*INCREASE,RESOLUTION*INCREASE))
            board = [[0 for x in range(RESOLUTION)] for y in range(RESOLUTION)]
            export = -1
        
        draw_grid(board)
        pg.display.update()

    pg.quit()


if __name__ == "__main__":
    main()
    df = create_df()
    df.to_csv("data/data.csv", index=False)
