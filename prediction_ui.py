import os
import pygame as pg
import numpy as np
import matplotlib.pyplot as plt
from package.utils import normalize_center_scale
import package.neunet as nn


# Initialize pygame
pg.init()
pg.display.set_caption("Dataset UI")

# Constants
SCREEN_RESOLUTION = 28 * 4
RESOLUTION = 28
PIXEL_SIZE = 4
BRUSH_SIZE = 14
N_DECIMALS = 3

def draw_board(board: list[list[int]], screen: pg.Surface):
    """
    Draw image pixel by pixel in the pygame screen.
    """
    for y in range(SCREEN_RESOLUTION):
        for x in range(SCREEN_RESOLUTION):
            g = int(255 * board[y][x])
            pg.draw.rect(
                screen,
                (g, g, g),
                (x * PIXEL_SIZE, y * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE),
            )


def to_npy(board: list[list[int]]):
    image = np.array(board).reshape(SCREEN_RESOLUTION, SCREEN_RESOLUTION)
    image = normalize_center_scale(image, RESOLUTION, int(SCREEN_RESOLUTION * 0.1))
    image = np.round(image, N_DECIMALS)
    image = image.reshape(784, 1)
    return image


def get_neighbours(x: int, y: int):
    """
    Returns a generator of the surrounding pixels of a pixel, that are inside the brush size,
    wich is a circle.
    """
    for i in range(-BRUSH_SIZE // 2, BRUSH_SIZE // 2 + 1):
        for j in range(-BRUSH_SIZE // 2, BRUSH_SIZE // 2 + 1):
            # Euclidean distance
            d = ((i) ** 2 + (j) ** 2) ** 0.5
            if 0 < d <= BRUSH_SIZE // 2:
                if 0 <= x + i < SCREEN_RESOLUTION and 0 <= y + j < SCREEN_RESOLUTION:
                    yield (y + j, x + i)


def main():
    """
    Create a screen to draw numbers and predict them.
    """

    # Initialize screen
    screen = pg.display.set_mode(
        (SCREEN_RESOLUTION * PIXEL_SIZE, SCREEN_RESOLUTION * PIXEL_SIZE)
    )

    # Initialize black board
    board = [[0 for _ in range(SCREEN_RESOLUTION)] for _ in range(SCREEN_RESOLUTION)]

    # Initialize clock with 120 FPS
    clock = pg.time.Clock()
    FPS = 120

    mnist = nn.NeuNet()
    mnist.load(model_name="mnist")

    run = True
    while run:

        # Events
        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False
            if event.type == pg.KEYUP:
                if event.key == pg.K_RETURN:
                    x = to_npy(board)
                    y = mnist.forward(x)
                    print(y.round(3))
                    print(np.argmax(y))
                if event.key == pg.K_BACKSPACE:
                    board = [
                        [0 for _ in range(SCREEN_RESOLUTION)]
                        for _ in range(SCREEN_RESOLUTION)
                    ]
                if event.key == pg.K_ESCAPE:
                    run = False

        # Paint brush
        mouse = pg.mouse.get_pressed()
        if any(mouse):
            j, i = pg.mouse.get_pos()
            j, i = j // PIXEL_SIZE, i // PIXEL_SIZE
            color = 1 if mouse[0] else 0
            board[i][j] = color
            for ci, cj in get_neighbours(j, i):
                board[ci][cj] = color


        # Draw
        draw_board(board, screen)
        pg.display.update()
        clock.tick(FPS)

    pg.quit()
    print("Bye!")


if __name__ == "__main__":
    main()
