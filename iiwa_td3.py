import numpy as np
import RobotDART as rd
import dartpy # OSX breaks if this is imported before RobotDART
from machin.frame.algorithms import TD3
from machin.utils.logging import default_logger as logger
import timeit
import pandas as pd
import gym
import copy
import torch as t
import torch.nn as nn


#class that creates iiwa enviroment resets its position and gives every step of every episode
class Env:
    def __init__(self, simu=rd.RobotDARTSimu(0.01), robot = rd.Iiwa(), graphics=rd.gui.Graphics(rd.gui.GraphicsConfiguration(1024, 768))):
########## Create simulator object ##########
       # time that each command runs for the robot
        self.dt = 0.01
        self.simu = simu

########## Load our new robot ##########
        self.robot =  robot
        self.simu.add_robot(robot)
        self.robot.set_actuator_types("servo")


########## Create Graphics ##########
# create graphics object with configuration and a window of 1024x768 resolution/size
        self.graphics = graphics
        self.simu.set_graphics(graphics)
        self.graphics.look_at([0., 3., 2.], [0., 0., 0.])

########## Fix robot to world frame ##########
        self.robot.fix_to_world()

########## Initial positions of iiwa ##########
        # set initial joint positions
        target_positions = copy.copy(self.robot.positions())
        target_positions[0] = np.pi
        target_positions[1] = -np.pi/2.0
        target_positions[2] = 0
        target_positions[3] = -np.pi/2.0
        target_positions[4] = 0
        target_positions[5] = np.pi/2.0

        self.robot.set_positions(target_positions)

#this  function resets the starting position of the iiwa at the start of each episode
    def reset(self):
        #initial position
        target_positions = copy.copy(self.robot.positions())
        target_positions[0] = np.pi
        target_positions[1] = -np.pi/2.0
        target_positions[2] = 0
        target_positions[3] = -np.pi/2.0
        target_positions[4] = 0
        target_positions[5] = np.pi/2.0

        self.robot.set_positions(target_positions)
        #commands are reseted to zero
        self.robot.set_commands([None,None,None,None,None,None,None])
        self.simu.step_world()
        
        theta=-np.pi+self.robot.positions()[0]
        #starting angle of iiwa`s lower body joint converted so that having a 90 degree angle from the ground gives value 0
        theta1=np.pi/2.0+self.robot.positions()[1]

        #starting angle of iiwa`s middle body joint giving 0 value when its paralell to the lower body
        theta2 = self.robot.positions()[3]

        #starting angle of iiwa`s higher body joint giving 0 value when its paralell to the middle body
        theta3 = self.robot.positions()[5]

        #angle velocity of every joint of iiwa
        thdot=self.robot.velocities()

         #starting state of iiwa
        state = np.array([np.cos(theta),np.cos(theta1),np.cos(theta2),np.cos(theta3) , np.sin(theta),np.sin(theta1), np.sin(theta2) ,np.sin(theta3),
        thdot[0] ,thdot[1] , thdot[2], thdot[3], thdot[4], thdot[5], thdot[6]], dtype=np.float32)
         #state is returned for the first state of the episode
        return state


    def step(self, action):
        terminal=False

        #immobilize iiwa`s rotation of the first body of iiwa from itself so that we make the learning process easier
        #ction[0]=0

        #commands for the robot
        self.robot.set_commands(action)
        self.simu.step_world()

        theta=-np.pi+self.robot.positions()[0]
 #new angle of iiwa`s lower body joint converted so that having a 90 degree angle from the ground gives value 0
        theta1=np.pi/2.0+self.robot.positions()[1]

    #new angle of iiwa`s middle body joint giving 0 value when its paralell to the lower body
        theta2 = self.robot.positions()[3]

    #new angle of iiwa`s higher body joint giving 0 value when its paralell to the middle body
        theta3 = self.robot.positions()[5]

  #angle velocity of every joint of iiwa
        thdot=self.robot.velocities()


      #reward for the action using the 3 angles, all velocities and all actions   
        reward = theta**2+theta1**2 + theta2**2 + theta3**2 + 0.1* thdot**2 + 0.001 * (action**2)

        newstate = np.array([np.cos(theta),np.cos(theta1),np.cos(theta2),np.cos(theta3) , np.sin(theta),np.sin(theta1), np.sin(theta2) ,np.sin(theta3),
        thdot[0] ,thdot[1] , thdot[2], thdot[3], thdot[4], thdot[5], thdot[6]], dtype=np.float32)

        return newstate, -reward.sum(), terminal, {}




#inialize iiwa enviroment and the algorithms parameters
env = Env()
observe_dim = 15
action_dim = 7
action_range = 2.5
max_episodes = 2000
max_steps = 500
noise_param = (0, 0.2)
noise_mode = "normal"
solved_reward = -2800
solved_repeat = 5


# model definition
#actor with 3 layers
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_range):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_dim)
        self.action_range = action_range

    def forward(self, state):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        a = t.tanh(self.fc3(a)) * self.action_range
        return a

#critic with 3 layers
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, state, action):
        state_action = t.cat([state, action], 1)
        q = t.relu(self.fc1(state_action))
        q = t.relu(self.fc2(q))
        q = self.fc3(q)
        return q


#main
if __name__ == "__main__":
#initialize actor critic and td3 algorithms
    actor = Actor(observe_dim, action_dim, action_range)
    actor_t = Actor(observe_dim, action_dim, action_range)
    critic = Critic(observe_dim, action_dim)
    critic_t = Critic(observe_dim, action_dim)
    critic2 = Critic(observe_dim, action_dim)
    critic2_t = Critic(observe_dim, action_dim)

    td3 = TD3(
        actor,
        actor_t,
        critic,
        critic_t,
        critic2,
        critic2_t,
        t.optim.Adam,
        nn.MSELoss(reduction="sum"),
    )
#counter for counting all the episodes
    n_iter = 1
     # array of every episode counter
    all_iter=[]
    # array for every episodes expected return
    all_exp_returns = []
    # array for every episodes runtime
    all_times = []
    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = 0

#every episode of the algorithm
    while episode < max_episodes:
        episode += 1
        total_reward = 0
        rewards = []
        terminal = False
        step = 0

    #starting state of episode and reseted iiwa
        state = t.tensor(env.reset(), dtype=t.float32).view(1, observe_dim)
        tmp_observations = []

        #timer
        start = timeit.default_timer()
    #every step of episode
        while step <= max_steps:
            #env.render()  #shows graphics
            step += 1
            with t.no_grad():
                #save previous state as old state
                old_state = state
                 #action of every step from td3 algorithm with noise
                action = td3.act_with_noise(
                    {"state": old_state}, noise_param=noise_param, mode=noise_mode
                    )
                act = np.array(action[0], dtype=np.float32)

                #new state, reward , terminal after using the action on iiwa
                state, reward, terminal, _ = env.step(act)
                state = t.tensor(state, dtype=t.float32).view(1, observe_dim)

                #save every reward of episode
                rewards.append(reward)
                #return of episode
                total_reward += reward
                
                  #observation of episode so that they are used on td3
                tmp_observations.append(
                    {
                        "state": {"state": old_state},
                        "action": {"action": action},
                        "next_state": {"state": state},
                        "reward": reward,
                        "terminal": terminal or step == max_steps,
                    }
                )
        #stop timer
        stop = timeit.default_timer()
        time = stop-start

        #store observations of episode
        td3.store_episode(tmp_observations)

         # show reward
        rewards = np.array(rewards)
        #array of every episode number
        all_iter.append(n_iter)
        #array of expected return of every episode
        all_exp_returns.append(np.mean(rewards.sum()))
        #array of run times of every episode
        all_times.append(time)

        #smoothed episode return from previous episodes so that we have a smooth stop on the algorithm and not a random one
        smoothed_total_reward = smoothed_total_reward * 0.5 + total_reward * 0.5
        logger.info(f"Episode {episode} smoothed_total_reward={smoothed_total_reward:.2f}")
        print("Iteration: {:6d}\tRuntime: {:6.4f}\tExpected return: {:6.2f}".format(n_iter, time, np.mean(rewards.sum())))
        n_iter += 1

  # after 20 episodes td3 is updated so that we have enough observations
        if episode > 20:
            for _ in range(step):
                td3.update()

    #5 times smoothed return is more than -300 algorithm is solved and it writes expected return and runtime on a csv file named iiwa_td3.csv
            if smoothed_total_reward > solved_reward:
                reward_fulfilled += 1
                if reward_fulfilled >= solved_repeat:
                    logger.info("Environment solved!")
                    df = pd.DataFrame({"episode" : all_iter, "expected_return" : all_exp_returns, "runtime" : all_times})
                    df.to_csv("iiwa_td3.csv", index=False)

                    exit(0)
            else:
                reward_fulfilled = 0   

        if (env.simu.step_world()):
            break