from GridWorld2D_agent import *
from GridWorld2D_env import *

ag = Agent()
print("initial Q-values ... \n")
print(ag.Qvalues)

ag.train(200)
print("latest Q-values ... \n")
print(ag.Qvalues)

ag.exp_rate = 0
print("Learned Q-values ... \n")
print(ag.Qvalues)
ag.test(10)

print(ag.scores)
print(len(ag.scores))
x = np.linspace(0, len(ag.scores),len(ag.scores))
y = ag.scores
plt.plot(x,y)
print(ag.Qvalues)
#plt.show()
