import gym
import numpy
from numpy import asarray
from numpy import savetxt
env = gym.make('CartPole-v0')
f = open("datasetModelInput.txt", "a")
f1 = open("datasetModelOutput.txt", "a")
for i_episode in range(1000):
    observation = env.reset()
    for t in range(20):
        env.render()

        #prendo azione random (la spinta)
        action = env.action_space.sample()

        #mi creo l'osservazione che mi serve, stato del sistema + input (la spinta)
        observation = numpy.append(observation, action)
        obs = ' '.join(map(str, observation))

        #mi salvo nel file degli "input" lo stato del sistema e l'ingresso (spinta a destra/sinistra), e vado a capo
        f.write(obs)
        f.write("\n")

        #lascio evolvere il sistema "di un passo"
        observation, reward, done, info = env.step(action)
        #l'osservazione qui è la risposta del sistema, del quale il suo stato iniziale, all'applicazione della spinta
        obs = ' '.join(map(str, observation))
        #mi salvo il risultato nel file degli "output"
        f1.write(obs)
        f1.write("\n")
        

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

f.close()
f1.close()
env.close()

