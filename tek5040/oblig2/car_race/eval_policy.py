
import sys
sys.path.append("../")
import numpy as np
import gym
import tensorflow as tf
import argparse
import time
from matplotlib import pyplot as plt
import matplotlib.animation as animation


from car_race import baselines
from car_race.common import preprocess
from car_race.videofig.videofig import videofig

#import pyvirtualdisplay
#_display = pyvirtualdisplay.Display(visible=False,  # use False with Xvfb
#                                    size=(1400, 900))
#_ = _display.start()

def show_episode(observations):

    def redraw_fn(i, axes):

        obs = observations[i]
        if not redraw_fn.initialized:
            redraw_fn.im = axes.imshow(obs, animated=True)
            redraw_fn.initialized = True
        else:
            redraw_fn.im.set_array(obs)

    redraw_fn.initialized = False

    videofig(len(observations), redraw_fn, play_fps=30)

def eval_policy(agent, num_steps, num_episodes, action_repeat=1):

    env = gym.make('CarRacing-v0')
    scores = []
    for i in range(num_episodes):
        # set seed so that we evaluate on same tracks each time
        env.seed(100*i)
        observation = env.reset()
        rewards = []
        highres_observations = []
        t = 0
        # TODO: Render more nice image...
        while True:
            highres_observations.append(env.render("rgb_array"))
            observation = preprocess(observation)
            action = agent(observation)
            action = np.array(action)[0]
            for _ in range(action_repeat):
                observation, reward, done, info = env.step(action)
                rewards.append(reward)
                t += 1
                if done:
                    break
                if num_steps >= 0 and t == num_steps:
                    break

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
            if num_steps >= 0 and t == num_steps:
                break

        score = sum(rewards)
        scores.append(score)
        if i == 0 or score > best_score:
            best_episode = highres_observations
            best_score = score

    env.close()

    return scores, best_episode

def main(agent, num_steps, num_episodes, action_repeat, render_best=True):

    scores, best_episode = eval_policy(agent, num_steps, num_episodes, action_repeat=action_repeat)

    print("min, max : (%g, %g)"  % (np.min(scores), np.max(scores)))
    print("median, mean : (%g, %g)" % (np.median(scores), np.mean(scores)))

    if render_best:
        show_episode(best_episode)

def parse_args():
    """Parse command line argument."""

    parser = argparse.ArgumentParser("Show policy on 'Car-Race-V0' task.")
    parser.add_argument("--num_steps", type=int, default=200,
                        help="Maximum number of steps for episode.")
    parser.add_argument("--num_episodes", type=int, default=100,
                        help="Number of steps to use for evaluation.")
    parser.add_argument("--action_repeat", type=int, default=1,
                        help="Number of steps to repeat each action.")
    parser.add_argument("--policy", default="random",
                        help="Either 'random' or 'straight' for baseline policies, or path to directory/file of saved model.")
    #parser.add_argument("--render", default=False, action='store_true',
    #                    help="Visualize or not.")

    return parser.parse_args()

def get_agent(policy):

    if policy == "random":
        return baselines.random
    elif policy == "straight":
        return baselines.straight
    else:
        agent = tf.keras.models.load_model(policy)
        return agent

if __name__ == '__main__':

    args = parse_args()

    agent = get_agent(args.policy)
    main(agent, args.num_steps, args.num_episodes, args.action_repeat)
