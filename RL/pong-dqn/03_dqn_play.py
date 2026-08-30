#!/usr/bin/env python3
import argparse
import collections

import gymnasium as gym
import numpy as np
import torch
from lib import dqn_model, wrappers

import time

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument(
        "-e",
        "--env",
        default=DEFAULT_ENV_NAME,
        help="Environment name to use, default=" + DEFAULT_ENV_NAME,
    )
    parser.add_argument(
        "-r",
        "--record",
        default=None,
        help="Directory to record video (if omitted, plays live in a window)",
    )
    args = parser.parse_args()

    if args.record:
        env = wrappers.make_env(args.env, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, video_folder=args.record)
    else:
        env = wrappers.make_env(args.env, render_mode="human")

    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n)
    state = torch.load(args.model, map_location=lambda stg, _: stg, weights_only=True)
    net.load_state_dict(state)

    state, _ = env.reset()
    total_reward = 0.0
    c: dict[int, int] = collections.Counter()

    while True:
        if not args.record:
            time.sleep(0.015)  # smooth ~60 FPS playback for human viewing
        state_v = torch.tensor(np.expand_dims(state, 0))
        q_vals = net(state_v).data.numpy()[0]
        action = int(np.argmax(q_vals))
        c[action] += 1
        state, reward, is_done, is_trunc, _ = env.step(action)
        total_reward += reward
        if is_done or is_trunc:
            break
    print("Total reward: %.2f" % total_reward)
    print("Action counts:", c)
    env.close()
