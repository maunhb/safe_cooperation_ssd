import ray
import sys
from ray import tune
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph
from ray.rllib.models import ModelCatalog
from ray.tune import run_experiments
from ray.tune.registry import register_env
import tensorflow as tf
from social_dilemmas.envs.state_matrix_env import MatrixEnv
from models.matrix_fc_net import FCNet
import numpy as np 

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'exp_name', 'prisoner_agents',
    'Name of the ray_results experiment directory where results are stored.')
tf.app.flags.DEFINE_string(
    'algorithm', 'A3C',
    'Name of the rllib algorithm to use.')
tf.app.flags.DEFINE_integer(
    'num_agents', 2,
    'Number of agent policies')
tf.app.flags.DEFINE_integer(
    'train_batch_size', 30000,
    'Size of the total dataset over which one epoch is computed.')
tf.app.flags.DEFINE_integer(
    'checkpoint_frequency', 20,
    'Number of steps before a checkpoint is saved.')
tf.app.flags.DEFINE_integer(
    'training_iterations', 20,
    'Total number of steps to train for')
tf.app.flags.DEFINE_integer(
    'num_cpus', 2,
    'Number of available CPUs')
tf.app.flags.DEFINE_integer(
    'num_gpus', 1,
    'Number of available GPUs')
tf.app.flags.DEFINE_boolean(
    'use_gpus_for_workers', False,
    'Set to true to run workers on GPUs rather than CPUs')
tf.app.flags.DEFINE_boolean(
    'use_gpu_for_driver', False,
    'Set to true to run driver on GPU rather than CPU.')
tf.app.flags.DEFINE_float(
    'num_workers_per_device', 1,
    'Number of workers to place on a single device (CPU or GPU)')

def setup(algorithm, train_batch_size, num_cpus, num_gpus,
          num_agents, use_gpus_for_workers=False, use_gpu_for_driver=False,
          num_workers_per_device=1):

    def env_creator(_):
        return MatrixEnv("prisoners")
    single_env = MatrixEnv("prisoners")

    env_name = "prisoners_env"
    register_env(env_name, env_creator)

    obs_space = single_env.observation_space
    act_space = single_env.action_space

    # Each policy can have a different configuration (including custom model)
    def gen_policy():
        return (PPOPolicyGraph, obs_space, act_space, {})

    # Setup PPO with an ensemble of `num_policies` different policy graphs
    policy_graphs = {}
    for i in range(num_agents):
        policy_graphs['agent-' + str(i)] = gen_policy()

    def policy_mapping_fn(agent_id):
        return agent_id

    # register the custom model
    model_name = "matrix_fc_net"
    ModelCatalog.register_custom_model(model_name, FCNet)

    agent_cls = get_agent_class(algorithm)
    config = agent_cls._default_config.copy()

    # information for replay
    config['env_config']['func_create'] = tune.function(env_creator)
    config['env_config']['env_name'] = env_name
    config['env_config']['run'] = algorithm

    # Calculate device configurations
    gpus_for_driver = int(use_gpu_for_driver)
    cpus_for_driver = 1 - gpus_for_driver
    if use_gpus_for_workers:
        spare_gpus = (num_gpus - gpus_for_driver)
        num_workers = int(spare_gpus * num_workers_per_device)
        num_gpus_per_worker = spare_gpus / num_workers
        num_cpus_per_worker = 0
    else:
        spare_cpus = (num_cpus - cpus_for_driver)
        num_workers = int(spare_cpus * num_workers_per_device)
        num_gpus_per_worker = 0
        num_cpus_per_worker = spare_cpus / num_workers

    # hyperparams
    config.update({
                "train_batch_size": FLAGS.train_batch_size,
                "horizon":  100, 
                "lr":  0.001,
                "num_workers": num_workers,
                "num_gpus": gpus_for_driver,  # The number of GPUs for the driver
                "num_cpus_for_driver": cpus_for_driver,
                "num_gpus_per_worker": num_gpus_per_worker,   # Can be a fraction
                "num_cpus_per_worker": num_cpus_per_worker,   # Can be a fraction
                "multiagent": {
                    "policy_graphs": policy_graphs,
                    "policy_mapping_fn": tune.function(policy_mapping_fn),
                },
                "model": {"custom_model": "matrix_fc_net", "use_lstm": True,
                          "lstm_cell_size": 128},

    })
    return algorithm, env_name, config


def main(unused_argv):
    ray.init(num_cpus=FLAGS.num_cpus, object_store_memory=50000000)
        
    alg_run, env_name, config = setup( FLAGS.algorithm,
                                      FLAGS.train_batch_size,
                                      FLAGS.num_cpus,
                                      FLAGS.num_gpus, FLAGS.num_agents,
                                      FLAGS.use_gpus_for_workers,
                                      FLAGS.use_gpu_for_driver,
                                      FLAGS.num_workers_per_device)

    if FLAGS.exp_name is None:
        exp_name = 'prisoners_' + FLAGS.algorithm
    else:
        exp_name = FLAGS.exp_name
    print('Commencing experiment', exp_name)

    run_experiments({
        exp_name: {
            "run": alg_run,
            "env": env_name,
            "stop": {
                "training_iteration": FLAGS.training_iterations
            },
            'checkpoint_freq': FLAGS.checkpoint_frequency,
            "config": config,
        }
    })


if __name__ == '__main__':
    tf.app.run(main)
