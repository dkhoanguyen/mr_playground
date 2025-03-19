import irsim  # initialize the environment with the configuration file
import irsim.world
import matplotlib.pyplot as plt
import numpy as np
import time

def main():
    # Simple setup
    env = irsim.make('world.yaml')
    plt.gcf().set_size_inches(12, 6) 
    
    #
    for i in range(300):  # run the simulation for 300 steps
        start = time.time()
        robot: irsim.world.ObjectBase
        # for robot in env.robot_list:
        #     robot.step(np.array([1,2]).reshape((2,1)))
        #
        env.step()
        env.render()  # render the environment
        
        if env.done():
            break  # check if the simulation is done
    env.end()  # close the environment

if __name__ == "__main__":
    main()
