import pygame as pg
import pandas as pd
import neunet as nn

pg.init()
L = 28
WIN = pg.display.set_mode((L*10,L*10+50))
pg.display.set_caption("Numbers")
font = pg.font.SysFont("comicsans", 30)


def draw_gameboard(board,sol, num):
    for y in range(L):
        for x in range(L):
            n = board[y][x]
            pg.draw.rect(WIN, (int(255*n),int(255*n),int(255*n)), (x*10, y*10, 10, 10))
    
    
    for i in range(10):
        color = (int(200*sol[i])+55,int(200*sol[i])+55,int(200*sol[i])+55)
        text = font.render(f"{i}", 1, color)
        WIN.blit(text, (i*28 + 5, L*10))
        if i == num:
            dot = font.render(".", 1, color)
            WIN.blit(dot, (i*28 + 10, L*10+ 10))
        else:
            dot = font.render(".", 1, (0,0,0))
            WIN.blit(dot, (i*28 + 10, L*10+ 10))
        
    
def main():
    board = [[0 for _ in range(L)] for _ in range(L)]
    clock = pg.time.Clock()
    export = -1
    run = True
    sol = [1,1,1,1,1,1,1,1,1,1]
    num = -1
    while run:
        clock.tick(120)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False

            if event.type == pg.KEYUP:
                if event.key == pg.K_RETURN:
                    if export == -1:
                        export = 0
                    if export == 1:
                        export = -1
                        board = [[0 for x in range(L)] for y in range(L)]
                        sol = [1,1,1,1,1,1,1,1,1,1]
                        num = -1
                    
        
        if export == -1:

            mouse = pg.mouse.get_pressed()
            if mouse[0]:
                x, y = pg.mouse.get_pos()
                x , y = x//10, y//10
                if x < L and y < L:
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
                if x < L and y < L:
                    board[y][x] = 0
                    if x < L-1:
                        board[y][x+1] = 0
                    if x > 0:
                        board[y][x-1] = 0
                    if y < L-1:
                        board[y+1][x] = 0
                    if y > 0:
                        board[y-1][x] = 0
        if export == 0:
            board_list = []
            for y in range(L):
                for x in range(L):
                    board_list.append(int(board[y][x]*255))
            df = pd.DataFrame(columns=['data', 'label'])
            df.loc[0] = [board_list, export]
            df.to_csv(f"numero.csv", index=False)
            export = 1
            sol = nn.test()
            num = sol.index(max(sol))

            total = sum(sol)
            for i in range(10):
                sol[i] /= total
        


        draw_gameboard(board, sol, num)
        pg.display.update()

    pg.quit()


if __name__ == "__main__":
    main()
