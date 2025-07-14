import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.layers import Lambda

# ========= Encoder Block =========
def encoder_block(x, filters, kernel_size=3):
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    skip = x  # For skip connection
    x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    return x, skip

# ========= Decoder Block =========
def decoder_block(x, skip, filters, kernel_size=3):
    x = layers.UpSampling2D(size=(2, 2), interpolation='nearest')(x)
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Concatenate()([x, skip])
    
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    return x

# ========= Full U-Net Model =========
def Unet(input_shape=(172, 79, 3)):
    # ----- Input + Padding -----
    inputs = Input(shape=input_shape)
    x = Lambda(lambda t: tf.pad(
        t,
        paddings=[[0, 0], [2, 2], [2, 3], [0, 0]],  # [batch, top/bottom, left/right, channels]
        mode='CONSTANT',
        constant_values=10
    ))(inputs)

    # ----- Encoder Path -----
    x, skip1 = encoder_block(x, 16)  # → 88 × 44
    x, skip2 = encoder_block(x, 32)  # → 44 × 22
    x, skip3 = encoder_block(x, 64)  # → 22 × 11

    # ----- Bottleneck -----
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # ----- Decoder 1 - for u -----
    x1 = decoder_block(x, skip3, 64)
    x1 = decoder_block(x1, skip2, 32)
    x1 = decoder_block(x1, skip1, 16)
    outputs1 = layers.Conv2D(1, 1, activation='linear', name='u_output')(x1)

    # ----- Decoder 2 - for v -----
    x2 = decoder_block(x, skip3, 64)
    x2 = decoder_block(x2, skip2, 32)
    x2 = decoder_block(x2, skip1, 16)
    outputs2 = layers.Conv2D(1, 1, activation='linear', name='v_output')(x2)

    # ----- Decoder 3 - for p -----
    x3 = decoder_block(x, skip3, 64)
    x3 = decoder_block(x3, skip2, 32)
    x3 = decoder_block(x3, skip1, 16)
    outputs3 = layers.Conv2D(1, 1, activation='linear', name='p_output')(x3)

    # ----- Model -----
    return Model(inputs, [outputs1, outputs2, outputs3], name='DeepCFD_U-Net')
