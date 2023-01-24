import os
import pygame as pg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Initialize pygame
pg.init()

# Multiple of 28
RESOLUTION = 28 * 10
screen = pg.display.set_mode((RESOLUTION, RESOLUTION))
pg.display.set_caption("Dataset UI")


def draw_grid(board):
    """Draw image pixel by pixel in the pygame screendow, augmenting the size of each pixel by INCREASE
    to make it more visible"""
    for y in range(RESOLUTION):
        for x in range(RESOLUTION):
            g = int(255 * board[y][x])
            pg.draw.rect(screen, (g, g, g), (x, y, 1, 1))


def save(label, board):
    """
    Save the image drawn in the pygame screendow in a png file in the data folder.
    Reduces the resolution of the image to 28x28
    """

    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists(f"data/{label}"):
        os.mkdir(f"data/{label}")

    # Find next image number
    next_image = 0
    while os.path.exists(f"data/{label}/{next_image}.png"):
        next_image += 1

    # Reduce resolution to 28x28 using mean
    img = [[0 for _ in range(28)] for _ in range(28)]
    WIDTH = RESOLUTION // 28
    HEIGHT = RESOLUTION // 28
    for y in range(28):
        for x in range(28):
            for i in range(HEIGHT):
                for j in range(WIDTH):
                    img[y][x] += board[y * HEIGHT + i][x * WIDTH + j]
            img[y][x] /= 28**2

    # Convert to image
    image = np.array(img).reshape(28, 28)

    # Save image
    plt.imsave(f"data/{label}/{next_image}.png", image, cmap="gray")


def get_neighbours(x, y, brush_size):
    """Get neighbours:
    Get the 8 neighbours of a pixel
    Returns: generator of the neighbours"""
    for i in range(-brush_size // 2, brush_size // 2 + 1):
        for j in range(-brush_size // 2, brush_size // 2 + 1):
            d = ((i) ** 2 + (j) ** 2) ** 0.5
            if 0 < d <= brush_size // 2:
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
        INTENSITY = 1
        SIZE = 35
        mouse = pg.mouse.get_pressed()
        if any(mouse):
            j, i = pg.mouse.get_pos()
            if mouse[0]:
                board[i][j] = 1
                for ci, cj in get_neighbours(j, i, SIZE):
                    # d = ((ci - i) ** 2 + (cj - j) ** 2) ** 0.5 / (SIZE // 2)
                    # board[ci][cj] = min(1, board[ci][cj] + INTENSITY / d)
                    board[ci][cj] = 1
            if mouse[2]:
                board[i][j] = 0
                for ci, cj in get_neighbours(j, i, SIZE):
                    board[ci][cj] = 0

        # Save image
        if label != -1:
            save(label, board)
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
