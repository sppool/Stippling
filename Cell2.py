import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy
import scipy.spatial


def update_coor():
    global n, fig, ax, move_rate, coor, border_min, border_max
    vor = scipy.spatial.Voronoi(coor)
    vertices_limit = np.clip(vor.vertices, border_min, border_max)
    mass_arr = []
    for i in range(n):
        vertice_lst = vor.regions[vor.point_region[i]]

        polygon = vertices_limit[vertice_lst]
        mass = get_mass(polygon)
        mass_arr.append(mass.copy())
    v = np.array(mass_arr) - coor[:-4]
    coor[:-4] += (move_rate * v)


def init():
    global border_min, border_max
    ax.set_xlim(border_min, border_max)
    ax.set_ylim(border_min, border_max)


def update(frame):
    p.set_data(coor[:, 0], coor[:, 1])
    update_coor()


def get_mass(polygon: np.array):  # shape (n, 2)
    polygon2 = np.roll(polygon, -1, axis=0)

    # Compute signed area of each triangle
    signed_areas = 0.5 * np.cross(polygon, polygon2)
    # Compute centroid of each triangle
    centroids = (polygon + polygon2) / 3.0
    # Get average of those centroids, weighted by the signed areas.
    mass = np.average(centroids, axis=0, weights=signed_areas)

    return mass


if __name__ == '__main__':
    border_min = 0
    border_max = 1
    move_rate = 0.5
    n = 1000  # 模擬數(太多很爽 但是會lag 吃光記憶體會當機 請緩慢增加)

    coor = np.random.uniform(0, 1, size=(n, 2))  # 位置座標
    coor = np.vstack((coor, [[border_min, border_min], [border_min, border_max], [
        border_max, border_min], [border_max, border_max]]))
    coor = coor.astype(np.float16)  # 節省資源

    fig_size = 7  # init圖形大小
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    p, = ax.plot('', '', 'or', ms=1.5)
    ani = FuncAnimation(fig=fig, func=update, frames=1,
                        init_func=init, interval=1, blit=False)  # fps=50
    plt.show()
