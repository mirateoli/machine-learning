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

x,y = np.array(ag.final_route).T
plt.plot(x,y)
plt.show()


# print(ag.scores)
# print(len(ag.scores))
# x = np.linspace(0, len(ag.scores),len(ag.scores))
# y = ag.scores
# plt.plot(x,y)
# print(ag.Qvalues)
# plt.show()
