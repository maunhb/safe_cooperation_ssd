
# two fully connected layers of size 32 each

import tensorflow as tf

from ray.rllib.models.misc import normc_initializer, flatten
from ray.rllib.models.model import Model
import tensorflow.contrib.slim as slim


class FCNet(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):

        inputs = input_dict["obs"]

        hiddens = [32, 32]
        with tf.name_scope("custom_net"):
            last_layer = inputs
            i = 1
            for size in hiddens:
                label = "fc{}".format(i)
                last_layer = slim.fully_connected(
                    last_layer,
                    size,
                    weights_initializer=normc_initializer(1.0),
                    activation_fn=tf.nn.relu,
                    scope=label)
                i += 1
            output = slim.fully_connected(
                last_layer,
                num_outputs,
                weights_initializer=normc_initializer(0.01),
                activation_fn=None,
                scope="fc_out")
            return output, last_layer
