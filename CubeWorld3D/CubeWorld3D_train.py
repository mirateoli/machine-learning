from CubeWorld3D_agent import *
from CubeWorld3D_env import *
import pyvista as pv

ag = Agent()
print("initial Q-values ... \n")
print(ag.Qvalues)

ag.train(1000)
#print("latest Q-values ... \n")
# print(ag.Qvalues)
# print("Best score:",ag.max_score)
# print("Best route:",ag.best_locations)


ag.exp_rate = 0
print("Learned Q-values ... \n")
print(ag.Qvalues)
ag.test(1)


###### PYVISTA VISUALIZING

force_float = False
obs_data = pv.PolyData(pv.wrap(np.array(ag.State.obstacles,dtype=float)))

box = pv.Box(bounds=(-0.45, 0.45, -0.45, 0.45, -0.4, 0.45))
obs_mesh = obs_data.glyph(scale=False, geom=box, orient=False)

print(ag.final_route)
route_data = ag.final_route
# route_data.insert(0,ag.State.start_location)
print(route_data)

edges = np.empty((0,2),dtype=int)

for i in range(len(route_data)-1):
    pair = [i, i + 1]
    edges = np.vstack((edges, pair))

# We must "pad" the edges to indicate to vtk how many points per edge
padding = np.empty(edges.shape[0], int) * 2
padding[:] = 2
edges_w_padding = np.vstack((padding, edges.T)).T
print(edges_w_padding)

route_mesh = pv.PolyData(route_data, edges_w_padding)
print(route_mesh)

colors = range(edges.shape[0])

pl = pv.Plotter()
pl.add_mesh(obs_mesh, color='r', opacity=0.5)
pl.add_mesh(route_mesh, render_lines_as_tubes=True,style='wireframe',line_width=10,color='blue',show_scalar_bar=False)
pl.background_color = 'white'
pl.show()


##### MATPLOTLIB VISUALIZING

# posx,posy,posz = np.array(ag.final_route).T
# obsx,obsy,obsz = np.array(ag.State.obstacles).T

# fig = plt.figure()
# ax1 = fig.add_subplot()
# ax1 = plt.axes(projection = '3d')

# ax1.plot(posx,posy,posz,linewidth=4)
# # ax1.scatter(obsx,obsy,obsz, s=100,c='r',marker='o')
# ax1.set_aspect('equal',adjustable='box')
# ax1.set_xlim(0,ag.State.xlim)
# ax1.set_ylim(0,ag.State.ylim)
# ax1.set_zlim(0,ag.State.zlim)

# # plt.xlim([0,ag.State.xlim])
# # plt.ylim([0,ag.State.ylim])

# plt.show()