import tensorflow as tf
from tensorflow.keras import layers, models



def simple_model(input_shape):

    height, width, channels = input_shape
    image = layers.Input(input_shape)
    x = layers.Conv2D(32, 5, strides=(2, 2), padding='same', activation='relu')(image)
    x = layers.Conv2D(64, 5, strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.Conv2D(1, 1, padding='same', activation=None)(x)
    # resize back into same size as regularization mask
    x = tf.image.resize(x, [height, width])
    x = tf.keras.activations.sigmoid(x)

    model = models.Model(inputs=image, outputs=x)

    return model


def conv2d_3x3(filters):
    conv = layers.Conv2D(
        filters, kernel_size=(3, 3), activation='relu', padding='same'
    )
    return conv

def deconv_3x3(filters):
    conv = layers.Conv2DTranspose(
        filters, kernel_size=(3, 3), strides=2, padding='same'
    )
    return conv

def max_pool():
    return layers.MaxPooling2D((2, 2), strides=2, padding='same')


def unet(input_shape):

    image = layers.Input(shape=input_shape)
    print(image)

    c1 = max_pool()(conv2d_3x3(8)(image))
    print(c1)
    c2 = max_pool()(conv2d_3x3(16)(c1))
    print(c2)
    c3 = max_pool()(conv2d_3x3(32)(c2))
    print(c3)
    c4 = max_pool()(conv2d_3x3(64)(c3))
    print(c4)

    p1 = max_pool()(conv2d_3x3(128)(c4))
    p1 = conv2d_3x3(128)(p1)
    print(p1)

    up4 = deconv_3x3(64)(p1)
    print(up4)
    cat4 = tf.keras.layers.concatenate([up4, c4])

    up3 = deconv_3x3(32)(cat4)
    cat3 = tf.keras.layers.concatenate([up3, c3])

    up2 = deconv_3x3(16)(cat3)
    cat2 = tf.keras.layers.concatenate([up2, c2])

    up1 = deconv_3x3(8)(cat2)
    cat1 = tf.keras.layers.concatenate([up1, c1])

    up0 = deconv_3x3(1)(cat1)

    probs = layers.Conv2D(1, kernel_size=(1, 1), activation='sigmoid')(up0)

    model = models.Model(inputs=image, outputs=probs)

    return model
