import tensorflow as tf
from tensorflow.keras import layers, Input, Model

def encoder_block(x, filters, kernel_size=3):
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    skip = x
    x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    return x, skip

def decoder_block(x, skip, filters, kernel_size=3):
    x = layers.Conv2DTranspose(filters, kernel_size=2, strides=2, padding='same')(x)
    # Handle potential mismatch by cropping the skip connection
    if x.shape[1] != skip.shape[1] or x.shape[2] != skip.shape[2]:
        height_diff = skip.shape[1] - x.shape[1]
        width_diff = skip.shape[2] - x.shape[2]
        skip = layers.Cropping2D(((0, height_diff), (0, width_diff)))(skip)
    x = layers.Concatenate()([x, skip])
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def Unet(input_shape=(172, 79, 3)):
    inputs = Input(shape=input_shape)

    # Encoder
    x, skip1 = encoder_block(inputs, 16)   # → ~ (86, 40)
    x, skip2 = encoder_block(x, 32)        # → ~ (43, 20)
    x, skip3 = encoder_block(x, 64)        # → ~ (22, 10)

    # Bottleneck
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Decoder 1 - for u
    x1 = decoder_block(x, skip3, 64)
    x1 = decoder_block(x1, skip2, 32)
    x1 = decoder_block(x1, skip1, 16)
    outputs1 = layers.Conv2D(1, 1, activation='linear')(x1)

    # Decoder 2 - for v
    x2 = decoder_block(x, skip3, 64)
    x2 = decoder_block(x2, skip2, 32)
    x2 = decoder_block(x2, skip1, 16)
    outputs2 = layers.Conv2D(1, 1, activation='linear')(x2)

    # Decoder 3 - for p
    x3 = decoder_block(x, skip3, 64)
    x3 = decoder_block(x3, skip2, 32)
    x3 = decoder_block(x3, skip1, 16)
    outputs3 = layers.Conv2D(1, 1, activation='linear')(x3)

    return Model(inputs, [outputs1, outputs2, outputs3], name='DeepCFD_U-Net')
