#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os, sys
import pickle

import gym
import ray
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from social_dilemmas.envs.state_matrix_env import MatrixEnv 
from social_dilemmas.envs.matrix_env import MatrixEnv as StatelessMatrixEnv
from social_dilemmas.matrix_fc_net import FCNet

import shutil
import utility_funcs
import numpy as np

num_steps = 100
num_trials = 100

matrix_game = "prisoners" # "staghunt" "routechoice"

state_env = "INPUT_TRAINED_ENV_CHECKPOINT_HERE" 
ModelCatalog.register_custom_model("matrix_fc_net", FCNet) 
register_env("env", lambda _: MatrixEnv(matrix_game)) 
register_env("stateless_env", lambda _: StatelessMatrixEnv(matrix_game)) 

ray.init(object_store_memory=50000000)

config_dir = os.path.dirname(state_env)
config_path = os.path.join(config_dir, "params.pkl")
if not os.path.exists(config_path):
    config_path = os.path.join(config_dir, "../params.pkl")
    #print('Exists in parent directory')
if not os.path.exists(config_path):
    raise ValueError(
        "Environment: Could not find params.pkl in either the checkpoint dir or "
        "its parent directory.")
with open(config_path, 'rb') as f:
    config = pickle.load(f)
if "num_workers" in config:
    config["num_workers"] = min(2, config["num_workers"])
if "ignore_worker_failures" in config:
    del config["ignore_worker_failures"]
    del config["metrics_smoothing_episodes"]
    del config["async_remote_worker_envs"]
if config["entropy_coeff"] < 0:
    config["entropy_coeff"] = 0.003
cls = get_agent_class("A3C")
env_agent = cls(env="env", config=config)
env_agent.restore(state_env)

def setup_agent(agent_dir):
    # Load configuration from file
    config_dir = os.path.dirname(agent_dir)
    config_path = os.path.join(config_dir, "params.pkl")
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.pkl")
        #print('Exists in parent directory')
    if not os.path.exists(config_path):
        raise ValueError(
            "Agent-0 Could not find params.pkl in either the checkpoint dir or "
            "its parent directory.")
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    if "num_workers" in config:
        config["num_workers"] = min(2, config["num_workers"])
    if "ignore_worker_failures" in config:
        del config["ignore_worker_failures"]
        del config["metrics_smoothing_episodes"]
        del config["async_remote_worker_envs"]
    if config["entropy_coeff"] < 0:
        config["entropy_coeff"] = 0.003
    agent_0 = cls(env="stateless_env", config=config)
    agent_0.restore(agent_dir)
    return agent_0


def rollout(agent_0, type_0, agent_1, type_1):
    agent_0_reward = 0
    agent_1_reward = 0
    cooperation_1 = np.zeros(num_steps)
    cooperation_2 = np.zeros(num_steps)
    if hasattr(agent_0, "local_evaluator"):
        multiagent_0 = agent_0.local_evaluator.multiagent
        if multiagent_0:
            policy_agent_mapping_0 = agent_0.config["multiagent"][
                "policy_mapping_fn"]
            mapping_cache_0 = {}
        policy_map_0 = agent_0.local_evaluator.policy_map
        state_init_0 = {p: m.get_initial_state() for p, m in policy_map_0.items()}
        use_lstm_0 = {p: len(s) > 0 for p, s in state_init_0.items()}
    if hasattr(agent_1, "local_evaluator"):
        multiagent_1 = agent_1.local_evaluator.multiagent
        if multiagent_1:
            policy_agent_mapping_1 = agent_1.config["multiagent"][
                "policy_mapping_fn"]
            mapping_cache_1 = {}
        policy_map_1 = agent_1.local_evaluator.policy_map
        state_init_1 = {p: m.get_initial_state() for p, m in policy_map_1.items()}
        use_lstm_1 = {p: len(s) > 0 for p, s in state_init_1.items()}

    env = env_agent.local_evaluator.env
    steps = 0

    while steps < (num_steps or steps + 1):
        state = env.reset()
        done = False
        while not done and steps < (num_steps or steps + 1):
            action_dict = {}
            for agent_id in state.keys():
                if agent_id == "agent-0":
                    if type_0 == "arctic":
                        a_state = round(state["agent-0"][0]*10)
                    else:
                        a_state = 0
                    policy_id = mapping_cache_0.setdefault(
                        "agent-0", policy_agent_mapping_0("agent-0"))
                    p_use_lstm = use_lstm_0[policy_id]
                    if p_use_lstm:
                        a_action, p_state_init, _ = agent_0.compute_action(
                            a_state,
                            state=state_init_0[policy_id],
                            policy_id=policy_id)
                        state_init_0[policy_id] = p_state_init
                    else:
                        a_action = agent_0.compute_action(
                            a_state, policy_id=policy_id)
                    if a_action > 1:
                        a_action = 1.0
                    elif a_action < 0:
                        a_action = 0.0
                    action_dict[agent_id] = a_action
                elif agent_id == "agent-1":
                    if type_1 == "arctic": 
                        a_state = round(state["agent-1"][0]*10)
                    else:
                        a_state = 0                     
                    policy_id = mapping_cache_1.setdefault(
                        "agent-0", policy_agent_mapping_1("agent-0"))
                    p_use_lstm = use_lstm_1[policy_id]
                    if p_use_lstm:
                        a_action, p_state_init, _ = agent_1.compute_action(
                            a_state,
                            state=state_init_1[policy_id],
                            policy_id=policy_id)
                        state_init_1[policy_id] = p_state_init
                    else:
                        a_action = agent_1.compute_action(
                            a_state, policy_id=policy_id)
                    if a_action > 1:
                        a_action = 1.0
                    elif a_action < 0:
                        a_action = 0.0
                    action_dict[agent_id] = a_action
            action = action_dict
            cooperation_1[steps] = 1.0 - action["agent-0"]
            cooperation_2[steps] = 1.0 - action["agent-1"]
            next_state, reward, done, info = env.step(action)
            done = done["__all__"]
            agent_0_reward += reward["agent-0"]
            agent_1_reward += reward["agent-1"]

            steps += 1
            state = next_state

    return cooperation_1, agent_0_reward, agent_1_reward

if __name__ == "__main__":
    ## To run a tournament, you must 
    ## input trained agent checkpoints and their types 

    ## Agent 1:
    AGENT_1 = "ENTER_CHECKPOINT_OF_TRAINED_AGENT"
    agent_1 = setup_agent(AGENT_1)
    type_1 = "arctic" # "adv", "baseline" etc

    ## Agent 2:
    AGENT_2 = "ENTER_CHECKPOINT_OF_TRAINED_AGENT"
    agent_2 = setup_agent(AGENT_2)
    type_2 = "arctic"

    ## tournament
    coop_file = open("./{}_{}_vs_{}_cooperation.csv".format(matrix_game,type_1,type_2), "a")
    coop_file.write("{},".format(type_1))
    for i in range(num_steps-1):
        coop_file.write("cooperation_{},".format(i+1))
    coop_file.write("cooperation_{}\n".format(num_steps))
    coop_file.close()
    rew_file = open("./{}_{}_vs_{}_rewards.csv".format(matrix_game,type_1,type_2), "a")
    rew_file.write("{},{},rewards_0,rewards_1\n".format(type_1,type_2))
    rew_file.close()
    for i in range(num_trials):
        c_1, r_1, r_2 = rollout(agent_1, type_1, agent_2, type_2)
        rew_file = open("./{}_{}_vs_{}_rewards.csv".format(matrix_game,type_1,type_2), "a")
        rew_file.write("{},{},{},{}\n".format(type_1,type_2,r_1,r_2))
        rew_file.close()
        coop_file = open("./{}_{}_vs_{}_cooperation.csv".format(matrix_game,type_1,type_2), "a")
        coop_file.write("{},".format(type_1))
        for i in range(len(c_1)-1):
            coop_file.write("{},".format(c_1[i]))
        coop_file.write("{}\n".format(c_1[-1]))
        coop_file.close()
    



