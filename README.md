
# Sequential Social Dilemma Games
This repo is an open-source implementation of Sequential Social Dilemma (SSD) multi-agent matrix game environments.
The implemented environments are structured to be compatible with OpenAIs gym environments (https://github.com/openai/gym) as well as RLlib's Multiagent Environment (https://github.com/ray-project/ray/blob/master/python/ray/rllib/env/multi_agent_env.py)

## Implemented Games

* **Prisoner's Dilemma**

* **Stag Hunt**

* **Route Choice**

# Setup instructions
* Create `causal` virtual environment: `conda env create -n causal environment.yml`
* Run `python setup.py develop`
* Activate your environment by running `source activate causal`, or `conda activate causal`.

To then set up the branch of Ray, install the ray wheel:
`pip install ray-0.6.5-cp36-cp36m-manylinux1_x86_64.whl`

Next, go to the rllib folder:
` cd ray/python/ray/rllib ` and run the script `python setup-rllib-dev.py`. This will copy the rllib folder into the pip install of Ray and allow you to use the version of RLlib that is in your local folder by creating a softlink. 

