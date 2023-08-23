from GridWorld2D_agent import *
from GridWorld2D_env import *

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


posx,posy = np.array(ag.final_route).T
obsx,obsy = np.array(ag.State.obstacles).T
fig = plt.figure()
ax1 = fig.add_subplot()

ax1.plot(posx,posy,linewidth=4)
ax1.scatter(obsx,obsy,s=800,c='r',marker='s')
ax1.set_aspect('equal',adjustable='box')

plt.xlim([0,ag.State.xlim])
plt.ylim([0,ag.State.ylim])
plt.show()


# print(ag.scores)
# print(len(ag.scores))
# x = np.linspace(0, len(ag.scores),len(ag.scores))
# y = ag.scores
# plt.plot(x,y)
# print(ag.Qvalues)
# plt.show()
