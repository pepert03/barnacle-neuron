import pygame as pg
import numpy as np
from package.utils import normalize_center_scale
import package.neunet as nn


# Initialize pygame
pg.init()
pg.display.set_caption("Dataset UI")
font = pg.font.SysFont("Comic Sans MS", 40)


# Constants
RESOLUTION = 28
SCREEN_RESOLUTION = 28 * 4
PIXEL_SIZE = 4
BRUSH_SIZE = 14
N_DECIMALS = 3

SCREEN_PIXELS = SCREEN_RESOLUTION * PIXEL_SIZE
FONT_W, FONT_H = font.size("0")
PRED_PADDING_X = 10
PRED_PADDING_Y = 5


def draw_board(board: list[list[int]], screen: pg.Surface, y_pred: list[int]):
    """
    Draw board in the pygame screen. Draw the prediction in the bottom of the screen.
    """
    # Draw board
    for y in range(SCREEN_RESOLUTION):
        for x in range(SCREEN_RESOLUTION):
            g = int(255 * board[y][x])
            pg.draw.rect(
                screen,
                (g, g, g),
                (x * PIXEL_SIZE, y * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE),
            )

    # Draw prediction
    vw = SCREEN_PIXELS - 2 * PRED_PADDING_X
    x_offset = (vw // 10 - FONT_W) // 2
    y = SCREEN_PIXELS + PRED_PADDING_Y
    for i in range(10):
        a = 50 + 205 * y_pred[i]
        text = font.render(str(i), True, (a, a, a))
        x = PRED_PADDING_X + (i * vw) // 10 + x_offset
        screen.blit(text, (x, y))


def to_npy(board: list[list[int]]):
    """
    Convert the board to a numpy array of shape (RESOLUTION**2, 1).
    """
    image = np.array(board).reshape(SCREEN_RESOLUTION, SCREEN_RESOLUTION)
    image = normalize_center_scale(image, RESOLUTION, int(SCREEN_RESOLUTION * 0.1))
    image = np.round(image, N_DECIMALS)
    image = image.reshape(RESOLUTION**2, 1)
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
        (SCREEN_PIXELS, SCREEN_PIXELS + FONT_H + 2 * PRED_PADDING_Y)
    )

    # Initialize black board
    board = [[0 for _ in range(SCREEN_RESOLUTION)] for _ in range(SCREEN_RESOLUTION)]

    # Initialize clock with 120 FPS
    clock = pg.time.Clock()
    FPS = 120

    # Initialize neural network
    mnist = nn.NeuNet()
    mnist.load(model_name="mnist")
    y_pred = np.ones(10)

    run = True
    while run:

        # Events
        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False
            if event.type == pg.KEYUP:
                if event.key == pg.K_RETURN:
                    x = to_npy(board)
                    y_pred = mnist.forward(x).round(N_DECIMALS)
                if event.key == pg.K_BACKSPACE:
                    y = np.ones(10)
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
            if i < SCREEN_PIXELS and j < SCREEN_PIXELS:
                j, i = j // PIXEL_SIZE, i // PIXEL_SIZE
                color = 1 if mouse[0] else 0
                board[i][j] = color
                for ci, cj in get_neighbours(j, i):
                    board[ci][cj] = color

        # Draw
        draw_board(board, screen, y_pred)
        pg.display.update()
        clock.tick(FPS)

    pg.quit()
    print("Bye!")


if __name__ == "__main__":
    main()
