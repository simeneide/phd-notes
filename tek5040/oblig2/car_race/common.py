import tensorflow as tf

def preprocess(observation):

    observation = tf.expand_dims(tf.cast(observation, tf.float32), 0)
    # change pixel values from [0, 255] to [-1, 1]
    observation = observation / 127.5 - 1

    return observation

class ActionEncoder(object):

    def __init__(self):

        # Try with discrete actions first
        self.actions = tf.constant([
            # [steer, gas, brake]
            [-1, 0, 0], # (turn left)
            [ 0, 0, 0], # (straight)
            [ 1, 0, 0], # (turn right)
            [ 0, 1, 0], # (gas)
            [ 0, 0, 1]  # (brake)
        ], dtype=tf.float32)

    @property
    def num_actions(self):
        return self.actions.shape[0]

    def index2action(self, index):
        """Simplified, only 5 actions..."""

        return tf.gather(self.actions, index)

    # This needed?
    def action2index(self, action):

        # [batch_size, 3] ==> [batch_size, 1, 3]
        action = tf.expand_dims(action, axis=1) # add num_actions dim
        # [num_actions, 3] ==> [1, num_actions, 3]
        actions = tf.expand_dims(self.actions, 0) # add batch size dim
        index = tf.argmin(tf.abs(action - actions), axis=-1, output_type=tf.dtypes.int32)

        return index
