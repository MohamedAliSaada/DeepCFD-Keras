#build the model as u net model with one encoder output only 
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Input, Model

def encoder_block(x, filters, kernel_size=3):
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    skip = x
    x = layers.MaxPooling2D()(x)
    return x, skip


def decoder_block(x, skip, filters, kernel_size=3):
    x = layers.UpSampling2D()(x)
    x = layers.Concatenate()([x, skip])
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def Unet_3(input_shape=(128, 128, 1)):
    inputs = Input(shape=input_shape)

    # Encoder
    x, skip1 = encoder_block(inputs, 16)
    x, skip2 = encoder_block(x, 32)
    x, skip3 = encoder_block(x, 64)

    # Bottleneck
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Decoder 1
    x1 = decoder_block(x, skip3, 64)
    x1 = decoder_block(x1, skip2, 32)
    x1 = decoder_block(x1, skip1, 16)
    # Output: 1 channel for (u)
    outputs1 = layers.Conv2D(1, 1, activation='linear')(x1)

    # Decoder 2
    x2 = decoder_block(x, skip3, 64)
    x2 = decoder_block(x2 , skip2, 32)
    x2 = decoder_block(x2 , skip1, 16)
    # Output: 1 channel for (v)
    outputs2 = layers.Conv2D(1, 1, activation='linear')(x2)

    # Decoder 3
    x3 = decoder_block(x, skip3, 64)
    x3 = decoder_block(x3, skip2, 32)
    x3 = decoder_block(x3, skip1, 16)
    # Output: 1 channel for (p)
    outputs3 = layers.Conv2D(1, 1, activation='linear')(x3)

    return Model(inputs, [outputs1,outputs2,outputs3], name='DeepCFD_U-Net')
