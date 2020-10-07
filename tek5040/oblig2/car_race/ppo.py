
import os
import sys
sys.path.append("../")
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import activations, layers, losses, optimizers
import gym
tf.keras.backend.set_floatx('float64')
from car_race.common import preprocess, ActionEncoder
from car_race.eval_policy import eval_policy
from car_race.networks import FeatureExtractor, PolicyNetwork, ValueNetwork


# Hold episode data
class EpisodeData:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.ts = [] # time step
        self.probs_old = []

def sample_episodes(env, policy_network, action_encoder, num_episodes, maxlen, action_repeat=1):

    episodes = []
    for i in range(num_episodes):
        episode = EpisodeData()
        observation = env.reset()
        for t in range(maxlen):
            observation = preprocess(observation)

            logits = policy_network.policy(observation)
            # remove num_samples dimension and batch dimension
            action = tf.random.categorical(logits, 1)[0][0]
            pi_old = activations.softmax(logits)[0]

            episode.observations.append(observation[0])
            episode.ts.append(np.float64(t))
            episode.actions.append(action.numpy())
            episode.probs_old.append(pi_old.numpy())

            reward = 0 # accumulate reward accross actions
            action = action_encoder.index2action(action).numpy()
            for _ in range(action_repeat):
                observation, r, done, info = env.step(action)
                reward = reward + r
                if done:
                    break

            episode.rewards.append(reward)
            if done:
                break

        episodes.append(episode)

    return episodes

def create_dataset(env, policy_network, value_network, action_encoder, num_episodes, maxlen, action_repeat, gamma):

    episodes = sample_episodes(env, policy_network, action_encoder, num_episodes, maxlen, action_repeat=action_repeat)

    dataset_size = 0
    for episode in episodes:

        # Could also get this when sampling episodes for efficiency
        # use predict?
        values = np.concatenate([value_network(np.expand_dims(o_t, 0), np.expand_dims(np.float64(maxlen)-t, 0)).numpy() for o_t, t in zip(episode.observations, episode.ts)])
        returns = calculate_returns(episode.rewards, gamma)
        advantages = returns - values

        episode.returns = returns
        episode.advantages = advantages
        dataset_size += len(episode.observations)

    slices = (
        tf.concat([e.observations for e in episodes], axis=0), # policy loss and value loss
        tf.concat([e.actions for e in episodes], axis=0), # policy loss
        tf.concat([e.advantages for e in episodes], axis=0), # policy loss
        tf.concat([e.probs_old for e in episodes], axis=0), # policy loss
        tf.concat([e.returns for e in episodes], axis=0), # value function loss
        tf.concat([e.ts for e in episodes], axis=0) # value function loss
    )
    dataset = tf.data.Dataset.from_tensor_slices(slices)
    dataset = dataset.shuffle(dataset_size)

    return dataset

def calculate_returns(rewards, gamma):
    """Calculate returns, i.e. sum of future discounted rewards, for episode.

    Args:
        rewards : array of shape [sequence_length], with elements
            being the immediate rewards [r_1, r_2, ..., r_T]
        gamma : discount factor, scalar value in [0, 1)

    Returns:
        returns: array of return for each time step, i.e. [g_0, g_1, ... g_{T-1}]
    """
    #rewards = np.random.randn((10))
    #gamma = 0.9
    ##    
    T = len(rewards)
    D = np.arange(0,T)
    returns = np.zeros(len(rewards), dtype=np.float64)

    for t in range(T):
        discount = gamma**D[:(T-t)]
        returns[t] = np.sum(discount*rewards[t:])

    return returns

def value_loss(target, prediction):
    """Calculate mean squared error loss for value function predictions.

    Args:
        target : tensor of shape [batch_size] with corresponding target values
        prediction : tensor of shape [batch_size] of predictions from value network

    Returns:
        loss : mean squared error difference between predictions and targets
    """
    #target = tf.ones((32,))
    #prediction = tf.ones((32,))*1.2
    # TODO: Implement value loss
    
    #loss = tf.math.reduce_mean(((target-prediction)**2))
    
    # The dummy output required a tensor of the same shape as target. but then its impossible to reduce to do mean. 
    # Outputting the squared difference for each datapoint
    
    return tf.reduce_mean((target-prediction)**2)

def policy_loss(pi_a, pi_old_a, advantage, epsilon):
    """Calculate policy loss as in https://arxiv.org/abs/1707.06347

    Args:
        pi_a : Tensor of shape [batch_size] with probabilities for actions under
            the current policy
        pi_old_a : Tensor of shape [batch_size] with probabilities for actions
            under the old policy
        advantage : Tensor of shape [batch_size] with estimated advantges for
            the actions (under the old policy)
        epsilon : clipping parameter, float value in (0, 1)

    Returns:
        loss : scalar loss value
    """

    u = pi_a/pi_old_a
    u_clip = tf.clip_by_value(u, 1-epsilon, 1+epsilon)
    loss = tf.reduce_mean(advantage * tf.minimum( u, u_clip))

    # TODO: implement policy loss
    #loss = tf.constant(0, dtype=tf.float64) # remove this line

    return loss

def estimate_improvement(pi_a, pi_old_a, advantage, t, gamma):
    """Calculate sample contributions to estimated improvement, ignoring changes to
    state visitation frequencies.

    Args:
        pi_a : Tensor of shape [batch_size] with probabilties for actions under
            the current policy
        pi_old_a : Tensor of shape [batch_size] with probabilities for actions
            under the old policy
        advantage : Tensor of shape [batch_size] with estimated advantges for
            the actions (under the old policy)
        t : Tensor of shape [batch_size], time step fo
        gamma : discount factor scalar value in [0, 1)
    Returns:
        Tensor of shape [batch_size] with estimated sample "contributions" to
        policy improvement.

    Note: in theory advantages*gamma^t should be close to zero, but may be
    different due to randomness or errors in our estimation. Subtracting this
    term seems to make more sense than not doing it.
    """

    # TODO: Implement this
    return tf.zeros(tf.shape(advantage), dtype=tf.float64) # remove this line

def estimate_improvement_lb(pi_a, pi_old_a, advantage, t, gamma, epsilon):
    """Estimate sample contributions to lower bound for improvement, ignoring
    changes to state visitation frequencies.

    Args:
        pi_a : Tensor of shape [batch_size] with probabilities for actions under
            the current policy
        pi_old_a : Tensor of shape [batch_size] with probabilities for actions
            under the old policy
        advantage : Tensor of shape [batch_size] with estimated advantges for
            the actions (under the old policy)
        epsilon : clipping parameter, float value in (0, 1)

    Note: in theory advantages*gamma^t should be close to zero, but may be
    different due to randomness or errors in our estimation. Subtracting this
    term seems to make more sense than not doing it.

    Returns:
        Tensor of shape [batch_size] with estimated sample "contributions" to
        lower bound of policy improvement.

    """

    # TODO: Implement this
    return tf.zeros(tf.shape(advantage), dtype=tf.float64) # remove this line

def entropy(p):
    """Entropy base 2, for each sample in batch."""

    log_p = tf.math.log(p + 1e-6) # add small number to avoid log(0)
    # use log2 for easier intepretation
    log2_p = log_p / np.log(2)

    # [batch_size x num_action] --> [batch_size]
    entropy = -tf.reduce_sum(p*log2_p, axis=-1)

    return entropy

def entropy_loss(pi):
    """Calculate entropy loss for action distributions.

    Args:
        pi : Tensor of shape [batch_size, num_actions], each element in the
            batch is a probability distribution over actions, conditoned on a
            state.

    Returns:
        scalar, average negative entropy for the distributions
    """
    # TODO: Implement this
    return -tf.reduce_mean(entropy(pi))
    #return tf.zeros(tf.shape(pi)[0], dtype=tf.float64) # remove this line

class Agent(tf.keras.models.Model):
    """Convenience wrapper around policy network, which returns *encoded*
    actions. Useful when running model for inference and evaluation.
    """

    def __init__(self, policy_network, action_encoder):
        super(Agent, self).__init__()
        self.policy_network = policy_network
        self.action_encoder = action_encoder

    def call(self, observation):
        index = self.policy_network(observation)
        action = self.action_encoder.index2action(index)
        return action

def main():

    run_name = "ppo_linear"
    base_dir = "train_out/" + run_name + "/"
    os.makedirs(base_dir, exist_ok=True)
    # initialize environment
    env = gym.make('CarRacing-v0')

    # initialize policy and value network
    action_encoder = ActionEncoder()
    feature_extractor = FeatureExtractor(conv=False, dense_hidden_units=0)
    policy_network = PolicyNetwork(feature_extractor, action_encoder.num_actions)
    policy_network._set_inputs(np.zeros([1, 96, 96, 3]))
    # Use to generate encoded actions (not just indices)
    agent = Agent(policy_network, action_encoder)
    agent._set_inputs(np.zeros([1, 96, 96, 3]))

    # use to keep track of best model
    mean_high = tf.Variable(0, dtype='float64', name='mean_high', trainable=False)

    # possibly share parameters with policy-network
    value_network = ValueNetwork(feature_extractor, hidden_units=0)

    iterations = 500
    K = 3
    num_episodes = 8
    maxlen_environment = 512
    action_repeat = 4
    maxlen = maxlen_environment // action_repeat # max number of actions
    batch_size = 32
    checkpoint_interval = 5
    eval_interval = 5
    eval_episodes = 8

    alpha_start = 1
    alpha_end = 0.0
    init_epsilon = 0.1
    init_lr = 0.5*10**-5
    optimizer = optimizers.Adam(init_lr)
    gamma = 0.99

    c1 = 0.1 # value function loss weight
    c2 = 0.001 # entropy bonus weight

    # Checkpointing
    checkpoint_path = base_dir + "checkpoints/"
    ckpt = tf.train.Checkpoint(
        policy_network=policy_network,
        value_network=value_network,
        mean_high=mean_high,
        optimizer=optimizer
    )
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        print("Restored weights from {}".format(ckpt_manager.latest_checkpoint))
        ckpt.restore(ckpt_manager.latest_checkpoint)
    else:
        print("Initializing random weights.")

    start_iteration = 0
    if ckpt_manager.latest_checkpoint:
        start_iteration = int(ckpt_manager.latest_checkpoint.split('-')[-1])

    # Summaries TensorBoard
    writer = tf.summary.create_file_writer(base_dir  + "summary/")
    kl_divergence = losses.KLDivergence(reduction=losses.Reduction.NONE)

    for iteration in range(start_iteration, iterations):

        # linearly decay alpha, change epsilon and learning rate accordingly
        alpha = (alpha_start-alpha_end)*(iterations-iteration)/iterations+alpha_end
        epsilon = init_epsilon * alpha # clipping paramter
        optimizer.learning_rate.assign(init_lr*alpha) # set learning rate

        ########################### Generate dataset ###########################
        start = time.time()
        dataset = create_dataset(env, policy_network, value_network,
                                 action_encoder, num_episodes, maxlen,
                                 action_repeat, gamma)
        print("Iteration: %d. Generated dataset in %f sec." %
              (iteration, time.time() - start))
        dataset = dataset.batch(batch_size)

        ############################## Training ################################
        start = time.time()

        # TODO: Implement one iteration of policy iteration. For each batch you
        # need to calculate the "loss" (the negative of what we want to
        # maximize), take the gradient of the loss with respect to the trainable
        # variables of both 'policy_network' and 'value_network', and update the
        # variables using the optimizer.
        for epoch in range(K):
            for batch in dataset:
                with tf.GradientTape() as tape:
                    observation, action, advantage, pi_old, value_target, t = batch
                    bs = len(observation)
                    pi_current = policy_network.policy(observation)
                    pi_current = activations.softmax(pi_current)
                    v = value_network(observation, maxlen-t)

                    action_idx = tf.transpose([tf.range(bs), action])
                    pi_old_a = tf.gather_nd(pi_old, action_idx)
                    pi_a = tf.gather_nd(pi_current, action_idx)
                    pl = policy_loss(pi_a, pi_old_a, advantage, epsilon) 
                    vl = value_loss(value_target, v)
                    el = entropy_loss(pi_a)
                    loss = pl + c1* vl + c2* el
                    #print(bs, loss, pl, vl, el)
                    parameters = []
                    parameters.extend(value_network.trainable_variables)
                    parameters.extend(policy_network.trainable_variables)
                    
                    grads = tape.gradient(loss, parameters)
                    optimizer.apply_gradients(zip(grads, parameters))
                    

        print("Iteration %d. Optimized surrogate loss in %f sec." %
              (iteration, time.time()-start))

        ############################## Summaries ###############################
        step = iteration + 1
        ratios, entropies, entropies_old, actions, advantages, kl_divs, diff, diff_lb = [], [], [], [], [], [], [], []
        for batch in dataset:
            obs, action, advantage, pi_old, value_target, t = batch
            action = tf.expand_dims(action, -1)
            logits = policy_network.policy(obs)
            pi = activations.softmax(logits)
            v = value_network(obs, np.float64(maxlen)-t)
            pi_a = tf.squeeze(tf.gather(pi, action, batch_dims=1), -1)
            pi_old_a = tf.squeeze(tf.gather(pi_old, action, batch_dims=1), -1)
            ratio = pi_a / pi_old_a

            kl_divs.append(kl_divergence(pi_old, pi))
            ratios.append(ratio)
            advantages.append(advantage)
            entropies.append(entropy(pi))
            entropies_old.append(entropy(pi_old))
            actions.append(tf.one_hot(tf.squeeze(action, -1), action_encoder.num_actions))
            diff.append(estimate_improvement(pi_a, pi_old_a, advantage, t, gamma))
            diff_lb.append(estimate_improvement_lb(pi_a, pi_old_a, advantage, t, gamma, epsilon))

        actions = tf.concat(actions, axis=0)
        action_frequencies = tf.reduce_sum(actions, axis=0) / tf.reduce_sum(actions)
        action_entropy = entropy(action_frequencies)
        ratios = tf.concat(ratios, axis=0)
        entropies = tf.concat(entropies, axis=0)
        entropies_old = tf.concat(entropies_old, axis=0)
        advantages = tf.concat(advantages, axis=0)
        diff = tf.concat(diff, axis=0)
        diff_lb = tf.concat(diff_lb, axis=0)
        kl_divs = tf.concat(kl_divs, axis=0)
        with writer.as_default():
            tf.summary.histogram("advantages", advantages, step=step)
            tf.summary.histogram("sample_improvements", diff, step=step)
            tf.summary.histogram("sample_improvements_lb", diff_lb, step=step)
            tf.summary.scalar("estimated_improvement",
                              tf.reduce_sum(diff)/num_episodes, step=step)
            tf.summary.scalar("estimated_improvement_lb",
                              tf.reduce_sum(diff_lb)/num_episodes, step=step)
            tf.summary.scalar("mean_advantage", tf.reduce_mean(advantages), step=step)
            tf.summary.histogram("entropy", entropies, step=step)
            tf.summary.histogram("prob_ratios", ratios, step=step)
            tf.summary.scalar("mean_entropy", tf.reduce_mean(entropies), step=step)
            tf.summary.scalar("mean_entropy_old", tf.reduce_mean(entropies_old), step=step)
            tf.summary.histogram("kl_divergence", kl_divs, step=step)
            tf.summary.scalar("mean_kl_divergence", tf.reduce_mean(kl_divs), step=step)
            tf.summary.scalar("kl_divergence_max", tf.reduce_max(kl_divs), step=step)
            tf.summary.scalar("action_entropy", action_entropy, step=step)
            tf.summary.scalar("action_freq_left", action_frequencies[0], step=step)
            tf.summary.scalar("action_freq_straight", action_frequencies[1], step=step)
            tf.summary.scalar("action_freq_right", action_frequencies[2], step=step)
            tf.summary.scalar("action_freq_gas", action_frequencies[3], step=step)
            tf.summary.scalar("action_freq_break", action_frequencies[4], step=step)

        ############################ Checkpointing #############################
        if step % checkpoint_interval == 0:
            print("Checkpointing model after %d iterations of training." % step)
            ckpt_manager.save(step)

        ############################# Evaluation ###############################
        if step % eval_interval == 0:
            start = time.time()

            scores, best_episode = eval_policy(
                agent, maxlen_environment, eval_episodes, action_repeat=action_repeat
            )
            m, M = np.min(scores), np.max(scores)
            median, mean = np.median(scores), np.mean(scores)

            print("Evaluated policy in %f sec. min, median, mean, max: (%g, %g, %g, %g)" %
                  (time.time() - start, m, median, mean, M))

            with writer.as_default():
                tf.summary.scalar("return_min", m, step=step)
                tf.summary.scalar("return_max", M, step=step)
                tf.summary.scalar("return_mean", mean, step=step)
                tf.summary.scalar("return_median", median, step=step)

            if mean > mean_high:
                print("New mean high! Old score: %g. New score: %g." %
                      (mean_high.numpy(), mean))
                mean_high.assign(mean)
                agent.save(os.path.join(base_dir, "high_score_model"))

    # Saving final model (in addition to highest scoring model already saved)
    # The model may be loaded with tf.keras.load_model(checkpoint_path + "agent")
    agent.save(checkpoint_path + "agent")

if __name__ == '__main__':
    #import pyvirtualdisplay
    #_display = pyvirtualdisplay.Display(visible=False,  # use False with Xvfb
    #                                    size=(1400, 900))
    #_ = _display.start()
    
    main()
