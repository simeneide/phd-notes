import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
import argparse


def _rows_cols(num_filters):

    num_filters = np.float32(num_filters)
    cols = int(np.ceil(np.sqrt(num_filters)))
    rows = (int(num_filters) + cols-1) // cols

    return rows, cols


def plot_filters(filters, max_filters_per_image=16):
    """[out_channels, kernel_height, kernel_width, 3]"""

    out_channels = filters.shape[0]
    h, w, c = filters.shape[1:]
    # space between filters
    s = 2

    num_images = int(np.ceil(out_channels / max_filters_per_image))
    if out_channels == 5:
        rows = cols = 3
    else:
        rows, cols = _rows_cols(out_channels)

    image = np.zeros((num_images, rows*h+(rows-1)*s, cols*w+(cols-1)*s, c))
    visualized = 0
    if out_channels == 5:
        i = 0
        r = 0; c = 1;
        image[i, r*(h+s):r*(h+s)+h,c*(w+s):c*(w+s)+w] = filters[3]
        r = 2; c = 1;
        image[i, r*(h+s):r*(h+s)+h,c*(w+s):c*(w+s)+w] = filters[4]
        r = 1
        for c in range(3):
            image[i, r*(h+s):r*(h+s)+h,c*(w+s):c*(w+s)+w] = filters[c]
        visualized = 5

    for i in range(num_images):
        for r in range(rows):
            for c in range(cols):
                if visualized >= out_channels:
                    break
                f = filters[i*rows*cols+r*cols+c]
                image[i, r*(h+s):r*(h+s)+h,c*(w+s):c*(w+s)+w] = f
                visualized += 1

    m, M = image.min(), image.max()
    image = np.round(255.0*(image - m)/(M - m + 1e-10)).astype(np.uint8)
    for im in image:
        plt.imshow(im)
        plt.show()

def show_conv_layer(layer):

    kernel = layer.weights[0]

    # [kernel_height, kernel_width, in_channels, out_channels]
    # --> [out_channels, kernel_height, kernel_width, in_channels]
    filters = tf.transpose(kernel, (3, 0, 1, 2))

    plot_filters(filters)

def show_dense_layer(layer):

    kernel = layer.weights[0]

    # [27648, units] --> [units, 27648] --> [units, 96, 96, 3]
    kernel = tf.reshape(tf.transpose(kernel), (-1, 96, 96, 3))

    plot_filters(kernel)

def plot_weights(agent):

    #has_conv_layers = not isinstance(agent.feature_extractor.layers[0], layers.Flatten)
    #has_hidden_dense = not isinstance(agent.feature_extractor.layers[-1], layers.Flatten)
    has_conv_layers = not agent.feature_extractor.layers[0].name == "flatten"
    has_hidden_dense = not agent.feature_extractor.layers[-1].name == "flatten"

    if has_conv_layers:
        show_conv_layer(agent.feature_extractor.layers[0])
    elif has_hidden_dense:
        show_dense_layer(agent.feature_extrator.layers[0])
    else:
        print("Showing weights for linear policy.")
        # TODO: show legend with action???
        show_dense_layer(agent.layers[-1])

def get_agent(policy):

    agent = tf.keras.models.load_model(policy)

    return agent

def main(policy):

    agent = get_agent(policy)
    #agent(tf.zeros([1, 96, 96, 3]))
    #agent.dense.build(tf.TensorShape([96, 96, 3]))
    #print(agent.summary())
    if hasattr(agent, "policy_network"):
        agent = agent.policy_network
    plot_weights(agent)

def parse_args():
    """Parse command line argument."""

    parser = argparse.ArgumentParser(
        "Show weights for policy network trained on 'Car-Race-V0'."
    )
    parser.add_argument("policy", help="Path to directory/file of saved model.")

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    main(args.policy)
