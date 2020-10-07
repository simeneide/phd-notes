import tensorflow as tf
from tensorflow.keras import activations, layers, optimizers
import gym


class FeatureExtractor(tf.keras.Model):

    def __init__(self, conv, dense_hidden_units, **kwargs):
        super(FeatureExtractor, self).__init__(**kwargs)

        self.conv = conv
        self.dense_hidden_units = dense_hidden_units

        if conv:
            self.conv1 = layers.Conv2D(16, kernel_size=8, strides=4, padding="SAME")
            self.conv2 = layers.Conv2D(32, kernel_size=4, strides=2, padding="SAME")

        self.flatten = layers.Flatten()
        if dense_hidden_units > 0:
            self.dense = layers.Dense(dense_hidden_units)

    def call(self, x, sample_action=True):

        if self.conv:
            # [96, 96] --> [24, 24]
            x = self.conv1(x)
            x = activations.relu(x)
            # [24, 24] --> [12, 12]
            x = self.conv2(x)
            x = activations.relu(x)
        x = self.flatten(x)
        if self.dense_hidden_units > 0:
            x = self.dense(x)
            x = activations.relu(x)

        return x

class PolicyNetwork(tf.keras.Model):
    """Policy network with discrete action space. Interpretation and encoding of
    actions are not handled here.
    """

    def __init__(self, feature_extractor, num_actions, **kwargs):
        super(PolicyNetwork, self).__init__(**kwargs)

        self.feature_extractor = feature_extractor
        self.dense = layers.Dense(num_actions)

    def policy(self, x):

        x = self.feature_extractor(x)
        logits = self.dense(x)

        return logits

    def _sample_action(self, logits):

        index = tf.random.categorical(logits, 1)
        # [batch_size, 1] ==> [batch_size]
        index = tf.squeeze(index, axis=-1)

        return index

    def call(self, x):

        logits = self.policy(x)
        action = self._sample_action(logits)

        return action

class ValueNetwork(tf.keras.Model):

    def __init__(self, feature_extractor=None, hidden_units=0, **kwargs):
        super(ValueNetwork, self).__init__(**kwargs)

        self.feature_extractor = feature_extractor
        self.hidden_units = hidden_units
        if hidden_units > 0:
            self.hidden = layers.Dense(hidden_units)

        self.value = layers.Dense(1)

    def call(self, observation, time_left):

        if self.feature_extractor is not None:
            # TODO: Better to project time_left, so that about same dimensions?
            tr = tf.expand_dims(time_left, -1)
            x = tf.concat([self.feature_extractor(observation), tr], -1)
        else:
            x = time_left

        if self.hidden_units:
            x = self.hidden(x)
            x = activations.relu(x)

        v = tf.squeeze(self.value(x), axis=-1)

        return v
