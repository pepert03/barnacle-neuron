import os
import pygame as pg
import numpy as np
import matplotlib.pyplot as plt

# Initialize pygame
pg.init()
pg.display.set_caption("Dataset UI")

# Constants
RESOLUTION = 28 * 10
BRUSH_SIZE = 35


def draw_board(board: list[list[int]], screen: pg.Surface):
    """
    Draw image pixel by pixel in the pygame screen.
    """
    for y in range(RESOLUTION):
        for x in range(RESOLUTION):
            g = int(255 * board[y][x])
            pg.draw.rect(screen, (g, g, g), (x, y, 1, 1))


def save(label: int | str, board: list[list[int]]):
    """
    Save the image drawn in the pygame screendow in a png file in the data folder.
    Reduces the resolution of the image to 28x28
    """

    # Create folders if they don't exist
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
                if 0 <= x + i < RESOLUTION and 0 <= y + j < RESOLUTION:
                    yield (y + j, x + i)


def main():
    """
    Create a screen to draw numbers and save them in the data folder,
    each number is saved in a folder with its label.
    """

    # Initialize screen
    screen = pg.display.set_mode((RESOLUTION, RESOLUTION))

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
        mouse = pg.mouse.get_pressed()
        if any(mouse):
            j, i = pg.mouse.get_pos()
            color = 1 if mouse[0] else 0
            board[i][j] = color
            for ci, cj in get_neighbours(j, i):
                board[ci][cj] = color

        # Save image
        if label != -1:
            save(label, board)
            board = [[0 for _ in range(RESOLUTION)] for _ in range(RESOLUTION)]
            label = -1

        # Draw
        draw_board(board, screen)
        pg.display.update()
        clock.tick(FPS)

    pg.quit()
    print("Bye!")


if __name__ == "__main__":
    main()
