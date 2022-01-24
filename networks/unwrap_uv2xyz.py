from util import obj_io
import numpy as np
import torch
from torchvision.utils import save_image
from matplotlib.path import Path
import cv2

def L2_dist(point_uv, point_tri):
    # piont_uv (n, 2) point_tri (3,2)
    d1 = np.power(np.power(point_uv - point_tri[0], 2).sum(1), 1 / 2)
    d2 = np.power(np.power(point_uv - point_tri[1], 2).sum(1), 1 / 2)
    d3 = np.power(np.power(point_uv - point_tri[2], 2).sum(1), 1 / 2)
    return np.vstack([d1,d2,d3]).T
    # return (n, 3)
def find_int_in_tri(poly, margin=0):
    xs, ys = zip(*poly)
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    x, y = np.meshgrid(np.arange(int(minx), int(maxx) + 1),
                       np.arange(int(miny), int(maxy) + 1))  # make a canvas with coordinates
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x, y)).T
    p = Path(poly)  # make a polygon
    grid = p.contains_points(points, radius=margin)

    return points[grid]

def cal_tri_area(tri):
    return 1/2 * abs(tri[0][0] * tri[1][1] + tri[1][0] * tri[2][1] + tri[2][0] * tri[0][1] - (tri[0][1] * tri[1][0] + tri[1][1]*tri[2][0] + tri[2][1] * tri[0][0]))


def uv_2_3dcoord_in_uv(uv, tri, tri3d):
    areas = []
    for i in range(3):
        tri_clone = tri.copy()
        tri_clone[i] = uv
        areas.append(cal_tri_area(tri_clone))
    areas = np.array(areas)
    weight = areas / areas.sum(axis=0)

    return weight@tri3d


obj_data =obj_io.load_obj_data('./main_obj_real.obj')
v = obj_data['v']
f = obj_data['f']
ft = obj_data['ft']
vt = obj_data['vt']

res = 1024

face2uv = np.array(list(map(lambda x: vt[x], ft))) * res
face2vertex = np.array(list(map(lambda x: v[x], f)))


uv_map = np.zeros((res, res, 3))
uv_map_margin = np.zeros((res, res, 3))

for i, tri in enumerate(face2uv):
    points = find_int_in_tri(tri, 0)
    #points_margin = find_int_in_tri(tri, 1)

    coord_vertex = face2vertex[i]
    for point in points:
        uv_map[-(point[1]+1), point[0]] = uv_2_3dcoord_in_uv(point, tri, coord_vertex)
    # for point in points:
    #     uv_map[-point[1], point[0]] = uv_2_3dcoord_in_uv(point, tri, coord_vertex)
    # for point in points_margin:
    #     if (uv_map_margin[-point[1], point[0]] == [0, 0, 0]).all():
    #         uv_map_margin[-point[1], point[0]] = uv_2_3dcoord_in_uv(point, tri, coord_vertex)



margin_list = np.arange(0.1, 3, 0.1)
for margin_val in margin_list:
    for i, tri in enumerate(face2uv):
        points_margin = find_int_in_tri(tri, margin_val)
        coord_vertex = face2vertex[i]
        for point in points_margin:
            if (uv_map[-(point[1]+1), point[0]] == [0, 0, 0]).all():
                uv_map[-(point[1]+1), point[0]] = uv_2_3dcoord_in_uv(point, tri, coord_vertex)


mask = uv_map == [0,0,0]
uv_map_final = uv_map_margin * mask + uv_map * (1-mask)

poly =face2uv[0]
xs, ys = zip(*poly)
minx, maxx = min(xs), max(xs)
miny, maxy = min(ys), max(ys)


x, y = np.meshgrid(np.arange(int(minx), int(maxx)+1), np.arange(int(miny), int(maxy)+1)) # make a canvas with coordinates
x, y = x.flatten(), y.flatten()
points = np.vstack((x,y)).T
p = Path(face2uv[0]) # make a polygon
grid = p.contains_points(points)
points[grid]



