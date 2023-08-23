from CubeWorld3D_agent import *
from CubeWorld3D_env import *

ag = Agent()
print("initial Q-values ... \n")
print(ag.Qvalues)

ag.train(5000)
#print("latest Q-values ... \n")
# print(ag.Qvalues)
# print("Best score:",ag.max_score)
# print("Best route:",ag.best_locations)


ag.exp_rate = 0
print("Learned Q-values ... \n")
print(ag.Qvalues)
ag.test(1)


posx,posy,posz = np.array(ag.final_route).T
obsx,obsy,obsz = np.array(ag.State.obstacles).T
fig = plt.figure()
ax1 = fig.add_subplot()
ax1 = plt.axes(projection = '3d')

ax1.plot(posx,posy,posz,linewidth=4)
ax1.scatter(obsx,obsy,obsz, s=100,c='r',marker='o')
ax1.set_aspect('equal',adjustable='box')
ax1.set_xlim(0,ag.State.xlim)
ax1.set_ylim(0,ag.State.ylim)
ax1.set_zlim(0,ag.State.zlim)

# plt.xlim([0,ag.State.xlim])
# plt.ylim([0,ag.State.ylim])

plt.show()