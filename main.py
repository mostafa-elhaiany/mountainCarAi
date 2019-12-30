import gym
import numpy as np
import matplotlib.pyplot as plt
env = gym.make("MountainCar-v0")


print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space.n)

discreteOsSize=[20]*len(env.observation_space.high)
discreteOsWindowSize = (env.observation_space.high - env.observation_space.low)/discreteOsSize

print(discreteOsWindowSize)


qTable= np.random.uniform(low=-2, high=0,size=(discreteOsSize+[env.action_space.n]))

episodeRewards=[]

aggrEpisodeRewards={'ep':[],'avg':[],'min':[],'max':[]}


print(qTable.shape)


learningRate=0.1
discount=0.95
episodes=25000
epsilon = 0.5
startEpsilonDecay=1
endEpsilonDecay=episodes//2
decayValue=epsilon/(endEpsilonDecay-startEpsilonDecay)

showEvery= 1

def getDiscreteState(state):
    discreteState= (state-env.observation_space.low)/discreteOsWindowSize
    return tuple(discreteState.astype(np.int))

for episode in range(episodes):
    episodeReward=0
    done=False
    discreteState=getDiscreteState(env.reset())
    while not done:
        action = np.argmax(qTable[discreteState])
        newState, reward, done,_ = env.step(action)
        episodeReward+=reward
        newDiscreteState= getDiscreteState(newState)
        env.render()
        if not done:
            maxFutureQ= np.max(qTable[newDiscreteState])
            currentQ=qTable[discreteState + (action,)]
            
            newQ= (1-learningRate) * currentQ + learningRate * (reward + discount * maxFutureQ)
            
            qTable[discreteState+(action,)]= newQ
            
        elif newState[0] >= env.goal_position:
            qTable[discreteState+(action,)]=1
            print(f'made it in episode {episode}')
            
        discreteState=newDiscreteState
    if(endEpsilonDecay>=episode>=startEpsilonDecay):
        epsilon-=decayValue
    episodeRewards.append(episodeReward)
    if(episode%showEvery==0):
        print(episode)
        avgReward= sum(episodeRewards[-showEvery:])/len(episodeRewards[-showEvery:])
        minReward=min(episodeRewards[-showEvery:])
        maxReward=max(episodeRewards[-showEvery:])
        aggrEpisodeRewards['ep'].append(episode)
        aggrEpisodeRewards['avg'].append(avgReward)
        aggrEpisodeRewards['min'].append(minReward)
        aggrEpisodeRewards['max'].append(maxReward)
        
        print(f"episode:{episode}, avg:{avgReward}, min:{minReward}, max:{maxReward}")

        
env.close()


plt.plot(aggrEpisodeRewards['ep'],aggrEpisodeRewards['avg'], label='avg')
plt.plot(aggrEpisodeRewards['ep'],aggrEpisodeRewards['min'], label='min')
plt.plot(aggrEpisodeRewards['ep'],aggrEpisodeRewards['max'], label='max')

plt.show()


