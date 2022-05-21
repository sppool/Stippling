import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation
from sklearn.metrics.pairwise import manhattan_distances  # 曼哈頓距離可以省大量計算
from sklearn.metrics.pairwise import euclidean_distances  # 歐幾里得距離


def update_coor_mov():
    global coor, mov
    d = euclidean_distances(coor, coor)
    d[d == 0] = 1  # 0會當機
    coor_ = ((coor[np.newaxis, :] - coor[:, np.newaxis]))
    coor_[d <= 0.001] = 0  # 太近不作用
    coor_[d >= 0.08] = 0  # 太遠不作用
    mov = mov - (coor_ / d[:, :, np.newaxis]**2 / 100000).sum(1)
    coor = coor + mov  # 更新位置
    mov *= 0.977

    # 碰撞轉換
    upper = coor >= 1
    lower = coor <= 0
    coor[upper] = 2 - coor[upper]  # 碰壁反彈
    coor[lower] = -coor[lower]  # 碰壁反彈
    # coor[upper] = 1  # 只在牆壁移動
    # coor[lower] = 0  # 只在牆壁移動

    # 碰撞向量轉換
    mov[upper] = -mov[upper]  # 碰壁反彈
    mov[lower] = -mov[lower]  # 碰壁反彈
    # mov[upper] = 0  # 只在牆壁移動
    # mov[lower] = 0  # 只在牆壁移動


def init():
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def update(frame):
    p.set_data(coor[:, 0], coor[:, 1])
    update_coor_mov()


n = 350  # 模擬數(太多很爽 但是會lag 吃光記憶體會當機 請緩慢增加)
coor = np.random.normal(0.5, 0.12, size=(n, 2))  # 位置座標
coor = coor.astype(np.float32)  # 節省資源
mov = np.zeros_like(coor)
w = np.ones((10, 10))
w[5:, 5:] = 2

fig_size = 5  # init圖形大小
fig, ax = plt.subplots(figsize=(fig_size, fig_size))
p, = ax.plot('', '', 'or', ms=2.5, alpha=0.7)
ani = FuncAnimation(fig=fig, func=update, frames=1,
                    init_func=init, interval=20, blit=False)  # fps=50
plt.show()
