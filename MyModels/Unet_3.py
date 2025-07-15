import tensorflow as tf
from tensorflow.keras import layers, Input, Model

# ========== Encoder Block ==========
def encoder_block(x, filters, kernel_size=3):
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    #x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    #x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    skip = x
    x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    return x, skip

# ========== Decoder Block ==========
def decoder_block(x, skip, filters, kernel_size=3):
    x = layers.UpSampling2D(size=(2, 2), interpolation='nearest')(x)
    x = layers.Concatenate()([x, skip])

    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    #x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    #x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    return x

# ========== Full U-Net With One Decoder and 3-channel Output ==========
def Unet_single_decoder(input_shape=(176, 80, 3)):
    inputs = Input(shape=input_shape)

    # ----- Encoder -----
    x, skip1 = encoder_block(inputs, 8)
    x, skip2 = encoder_block(x, 16)
    x, skip3 = encoder_block(x, 32)

    # ----- Bottleneck -----
    x = layers.Conv2D(32, 3, padding='same')(x)
    #x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # ----- Decoder -----
    x = decoder_block(x, skip3, 32)
    x = decoder_block(x, skip2, 16)
    x = decoder_block(x, skip1, 8)

    # ----- Final Output -----
    outputs = layers.Conv2D(1, 1, activation='linear', name='output')(x)

    return Model(inputs, outputs, name='DeepCFD_U-Net_single_decoder')
