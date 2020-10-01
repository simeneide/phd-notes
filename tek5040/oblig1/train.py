#%%
import glob
import os
import argparse
import time
from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow.keras import losses, metrics, optimizers

import segmentation_models


############################  Data generation ##################################
HEIGHT = 256
WIDTH = 256

def generator_for_filenames(*filenames):
    """
    Wrapping a list of filenames as a generator function
    """
    def generator():
        for f in zip(*filenames):
            yield f
    return generator


def preprocess(image, segmentation, random_augmentation=False):
    """ A preprocess function the is run after images are read. Here you can do
    augmentation and other processesing on the images.
    """

    # Set images size to a constant
    image = tf.image.resize(image, [HEIGHT, WIDTH])
    segmentation = tf.image.resize(segmentation, [HEIGHT, WIDTH],
                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.cast(image, tf.float32) / 255
    segmentation = tf.cast(segmentation, tf.int32)

    if random_augmentation:
        raise NotImplementedError("You need to implement this if you want it.")

    return image, segmentation


def read_image_and_segmentation(img_f, seg_f):
    """Read images from file using tensorflow and convert the segmentation to
    appropriate formate.

    Args:
      img_f: filename for image
      seg_f: filename for segmentation

    Returns:
      Image and segmentation tensors
    """
    img_reader = tf.io.read_file(img_f)
    seg_reader = tf.io.read_file(seg_f)
    img = tf.image.decode_png(img_reader, channels=3)
    seg = tf.image.decode_png(seg_reader)[:, :, 2:]
    seg = tf.where(seg > 0, tf.ones_like(seg), tf.zeros_like(seg))
    return img, seg


def kitti_dataset_from_filenames(image_names, segmentation_names, preprocess=preprocess, batch_size=8, shuffle=True):
    """Convert a list of filenames to tensorflow images.

    Args:
      image_names: image filenames
      segmentation_names: segmentation filenames
      preprocess: A function that is run after the images are read, the takes
        image and segmentation as input
      batch_size: The batch size returned from the function

    Returns:
      Tensors with images and corresponding segmentations
    """
    dataset = tf.data.Dataset.from_generator(
        generator_for_filenames(image_names, segmentation_names),
        output_types=(tf.string, tf.string),
        output_shapes=(None, None)
    )

    if (shuffle):
        dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.map(read_image_and_segmentation)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch_size)

    return dataset


def kitti_image_filenames(dataset_folder, training=True):
    sub_dataset = 'training' if training else 'testing'
    segmentation_names = glob.glob(os.path.join(dataset_folder, sub_dataset, 'gt_image_2', '*road*.png'),
                                   recursive=True)
    image_names = [f.replace('gt_image_2', 'image_2').replace('_road_', '_') for f in segmentation_names]
    return image_names, segmentation_names


def vis_mask(image, mask, alpha=0.4):
    """Visualize mask on top of image, blend using 'alpha'."""

    # Note that as images are normalized, 1 is max-value
    red = tf.zeros_like(image) + tf.constant([1, 0, 0], dtype=tf.float32)
    vis = tf.where(mask, alpha*image+(1-alpha)*red, image)

    return vis

def get_train_data(subset_indices, batch_size):

    # Getting filenames from the kitti dataset
    image_names, segmentation_names = kitti_image_filenames('data_road')

    preprocess_train = partial(preprocess, random_augmentation=False)

    # Get image tensors from the filenames
    train_data = kitti_dataset_from_filenames(
        [image_names[idx] for idx in subset_indices],
        [segmentation_names[idx] for idx in subset_indices],
        preprocess=preprocess_train,
        batch_size=batch_size
    )
    return train_data

def get_val_data(subset_indices, batch_size):

    # Getting filenames from the kitti dataset
    image_names, segmentation_names = kitti_image_filenames('data_road')

    preprocess_val = preprocess

    # Get the validation tensors
    val_data = kitti_dataset_from_filenames(
        [image_names[idx] for idx in subset_indices],
        [segmentation_names[idx] for idx in subset_indices],
        batch_size=batch_size,
        preprocess=preprocess_val,
        shuffle=False
    )
    return val_data

#############################  Loss and metric  ################################

def get_loss_fn():
    loss_fn = losses.BinaryCrossentropy(from_logits=False)
    return loss_fn

# Note that the functionality below we could have used the functionaliry of
# metrics.BinaryAccuracy(threshold=0.5).
def preds_evaluated(y_true, y_pred):
    """Evaluate all predicitions as correct or incorrect, where we choose the
    class label of the prediction to be 1 if the probability is 0.5 or higher.
    If we take the average of these evaluated predictions we get the accuracy.

    Args:
      y_true : int tensor of shape [batch_size, height, width, 1] with ground
        truth labels.
      y_pred : float tensor of shape [batch_size, height, width 1] tensor of
        probabilities in [0, 1]

    Returns:
      float array of shape [batch_size*height*width] where the value is 1 if
      the corresponsing prediction of the pixel has the correct value, or 0
      otherwise
    """
    # set label to 1 if probabiliy is greaater than 0.5
    y_pred = tf.where(y_pred >= 0.5, 1, 0)
    correct = y_pred == y_true
    # [batch_size, height, width, 1] ==> [batch_size*height*width]
    correct = tf.reshape(correct, [-1])
    correct = tf.cast(correct, tf.float32)

    return correct

##################  Optimizer with learning rate schedule ######################

def get_optimizer():
    # Just constant learning rate schedule
    optimizer = optimizers.Adam(lr=1e-4)
    return optimizer

def main(train_dir, model_name=None):

    train_epochs = 12
    train_batch_size = 4
    val_batch_size = 2*train_batch_size

    # Divide into train and val
    indices = np.arange(287)
    # Set seed so that we draw same images each time.
    np.random.seed(123)
    np.random.shuffle(indices)

    train_data = get_train_data(indices[:272], train_batch_size)
    val_data = get_val_data(indices[272:], val_batch_size)
    input_shape = (HEIGHT, WIDTH, 3)
    if model_name=="conv":
        model = segmentation_models.simple_model(input_shape)
    elif model_name =="unet":
        model = segmentation_models.unet(input_shape)
    loss_fn = get_loss_fn()
    metric_fn = preds_evaluated
    optimizer = get_optimizer()

    # Used to running averages for summaries
    train_accuracy = metrics.Mean()
    train_loss = metrics.Mean()
    val_accuracy = metrics.Mean()
    val_loss = metrics.Mean()

    def train_step(image, y):
        """Updates model parameters, as well as `train_loss` and `train_accuracy`.

        Args:
          image: float tensor of shape [batch_size, height, width, 3]
          y: ground truth float tensor of shape [batch_size, height, width, 1]

        Returns:
          float tensor of shape [batch_size, height, width, 1] with probabilties
        """
        with tf.GradientTape() as tape:            
            y_pred = model(image, training=True)
            loss = loss_fn(y, y_pred)
    
            grads = tape.gradient(loss,model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            train_loss.update_state(loss)
            train_accuracy.update_state(metric_fn(y,y_pred))

        # TODO: Implement

        return y_pred

    def val_step(image, y):
        """Update `val_loss` and `val_accuracy`.

        Args:
          image: float tensor of shape [batch_size, height, width, 3]
          y: ground truth float tensor of shape [batch_size, height, width, 1]

        Returns:
          float tensor of shape [batch_size, height, width, 1] with probabilties
        """
        y_pred = model(image, training=False)
        loss = loss_fn(y, y_pred)
        val_loss.update_state(loss)
        val_accuracy.update_state(metric_fn(y,y_pred))

        return y_pred

    print("Summaries are written to '%s'." % train_dir)
    train_writer = tf.summary.create_file_writer(
        os.path.join(train_dir, "train"), flush_millis=3000)
    val_writer = tf.summary.create_file_writer(
        os.path.join(train_dir, "val"), flush_millis=3000)
    summary_interval = 10

    step = 0
    start_training = start = time.time()
    for epoch in range(train_epochs):

        print("Training epoch: %d" % epoch)
        for image, y in train_data:
            y_pred = train_step(image, y)
            step += 1

            # summaries to terminal
            if step % summary_interval == 0:
                duration = time.time() - start
                print("step %3d. sec/batch: %.3f. Train loss: %g" % (
                    step, duration/summary_interval, train_loss.result().numpy()))
                start = time.time()

        # write summaries to TensorBoard
        with train_writer.as_default():
            tf.summary.scalar("loss", train_loss.result(), step=epoch+1)
            tf.summary.scalar("accuracy", train_accuracy.result(), step=epoch+1)
            vis = vis_mask(image, y_pred >= 0.5)
            tf.summary.image("train_image", vis, step=epoch+1)

        # reset metrics
        train_loss.reset_states()
        train_accuracy.reset_states()

        # Do validation after each epoch
        for i, (image, y) in enumerate(val_data):
            y_pred = val_step(image, y)

            # Visualize all images in the validation set.
            with val_writer.as_default():
                vis = vis_mask(image, y_pred >= 0.5)
                tf.summary.image("val_image_batch_%d" % i, vis, step=epoch+1, max_outputs=val_batch_size)

        with val_writer.as_default():
            tf.summary.scalar("loss", val_loss.result(), step=epoch+1)
            tf.summary.scalar("accuracy", val_accuracy.result(), step=epoch+1)
        val_loss.reset_states()
        val_accuracy.reset_states()

    print("Finished training %d epochs in %g minutes." % (
        train_epochs, (time.time() - start_training)/60))
    # save a model which we can later load by tf.keras.models.load_model(model_path)
    model_path = os.path.join(train_dir, "model.h5")
    print("Saving model to '%s'." % model_path)
    model.save(model_path)

def parse_args():
    """Parse command line argument."""
    try:
        parser = argparse.ArgumentParser("Train segmention model on Kitti dataset.")
        parser.add_argument("train_dir", help="Directory to put logs and saved model.")
        parser.parse_args()
    except:
        args = {'train_dir' : "tensorboard"}
    return 


if __name__ == '__main__':

    #args = parse_args()
    train_dir = "tensorboard"
    model_name = "conv"
    main(train_dir = f"{train_dir}/{model_name}", model_name = model_name)




