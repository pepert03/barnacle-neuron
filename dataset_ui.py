import os
import pygame as pg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


pg.init()
RESOLUTION = 28
INCREASE = 10
screen = pg.display.set_mode((RESOLUTION * INCREASE, RESOLUTION * INCREASE))
pg.display.set_caption("Dataset UI")


def draw_grid(board):
    """Draw image pixel by pixel in the pygame screendow, augmenting the size of each pixel by INCREASE
    to make it more visible"""
    for y in range(RESOLUTION):
        for x in range(RESOLUTION):
            g = int(255 * board[y][x])
            pg.draw.rect(
                screen, (g, g, g), (x * INCREASE, y * INCREASE, INCREASE, INCREASE)
            )


def save(label):
    """
    Save the image drawn in the pygame screendow in a png file in the data folder
    """

    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists(f"data/{label}"):
        os.mkdir(f"data/{label}")

    i = 0
    while not os.path.exists(f"data/{label}/{i}.png"):
        i += 1
    pg.image.save(screen, f"data/{label}/{i}.png")


def create_df():
    """
    Create a dataframe with the data and the label of each image
    """

    df = pd.DataFrame(columns=["data", "label"])
    for i in range(10):
        folder = str(i)
        for file in os.listdir("data/" + folder):
            img = plt.imread("data/" + folder + "/" + file)
            gray = (
                (np.mean(img, axis=2).reshape(1, -1) * 255).astype(np.uint8).tolist()[0]
            )
            df.loc[len(df)] = [gray, folder]

    return df


def get_neighbours(x, y):
    """Get neighbours:
    Get the 8 neighbours of a pixel
    Returns: generator of the neighbours"""
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            if 0 <= x + i < RESOLUTION and 0 <= y + j < RESOLUTION:
                yield (y + j, x + i)


def main():
    """
    Create a screendow to draw numbers and save them in the data folder
    """

    # Initialize black board
    board = [[0 for _ in range(RESOLUTION)] for _ in range(RESOLUTION)]

    # Image label
    label = -1

    # Initialize clock with 60 FPS
    clock = pg.time.Clock()
    FPS = 120

    run = True
    while run:

        # Events
        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False
            if event.type == pg.KEYUP:
                if pg.K_0 <= event.key <= pg.K_9:
                    label = event.key - pg.K_0
                if event.key == pg.K_RETURN:
                    board = [[0 for _ in range(RESOLUTION)] for _ in range(RESOLUTION)]
                if event.key == pg.K_BACKSPACE:
                    board = [[0 for _ in range(RESOLUTION)] for _ in range(RESOLUTION)]
                if event.key == pg.K_ESCAPE:
                    run = False

        # Paint brush
        intensity = 0.25
        mouse = pg.mouse.get_pressed()
        if any(mouse):
            x, y = pg.mouse.get_pos()
            j, i = x // INCREASE, y // INCREASE
            if mouse[0]:
                board[i][j] = 1
                for ci, cj in get_neighbours(j, i):
                    d = abs(ci - i) + abs(cj - j)
                    board[ci][cj] = min(1, board[ci][cj] + intensity / (2 * d))
            if mouse[2]:
                board[i][j] = 0
                for ci, cj in get_neighbours(j, i):
                    board[ci][cj] = 0

        # Save image
        if label != -1:
            save(label)
            board = [[0 for _ in range(RESOLUTION)] for _ in range(RESOLUTION)]
            label = -1

        # Draw
        draw_grid(board)
        pg.display.update()
        clock.tick(FPS)

    pg.quit()
    print("Bye!")


if __name__ == "__main__":
    main()
