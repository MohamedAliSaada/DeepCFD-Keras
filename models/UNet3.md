# U-Net with Multi-Decoder Paths as Keras Custom Layer (double checked)

This repository provides a custom **UNet3** implementation in Keras using modular encoder, neck, and multi-path decoder blocks. The architecture mimics and extends the classical U-Net structure with skip connections and **multiple output decoder paths**—ideal for multi-channel segmentation, regression, or scientific modeling tasks (e.g., CFD, multi-field medical imaging).

---

## 🔧 Features

* Encoder and neck blocks using sequential `Conv2D` layers
* **Multiple decoder branches** (one per output channel)
* Skip connections for rich feature fusion, classic U-Net style
* Customizable kernel size, filter depth, activation functions, and batch normalization
* Each decoder path is an independent model ending with a 1-channel output
* Layer inspection via `My_layers_is()` method

---

## 📦 Installation

No special installation required. Just place the file in your TensorFlow project directory.

---

## 🚀 Usage

### 1. Import the Layer

```python
from unet3 import UNet3
```

### 2. Create a UNet3 instance

```python
model = UNet3(
    filters=[16, 32, 64],           # encoder/decoder filters
    kernel_size=(3, 3),             # must be odd values
    batch_norm=True,                # toggle for BatchNormalization
    activation="relu",             # activation function for hidden layers
    final_activation="sigmoid",    # output layer activation
    out_channels=3                  # number of output channels/decoder branches
)
```

### 3. Build and Use the Model

```python
import tensorflow as tf
input_tensor = tf.keras.Input(shape=(128, 128, 1))
output_tensor = model(input_tensor)
full_model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
```

**Expected Output Shape:**
If `input_tensor` has shape `(128, 128, 1)` and `out_channels=3`, the final output will be:

```python
output_tensor.shape == (128, 128, 3)
```

### 4. Compile and Train

```python
full_model.compile(optimizer='adam', loss='mse')
# full_model.fit(x_train, y_train, epochs=10, batch_size=32)
```

### 5. Inspect the Layers

```python
model.My_layers_is()
```

Example Output:

```
Encoder Layers:
   Conv2D
   BatchNormalization
   Activation
   Conv2D
   BatchNormalization
   Activation
   MaxPooling2D
   ...
Neck Layers:
   Conv2D
   BatchNormalization
   Activation
   Conv2D
   BatchNormalization
   Activation
Decoder Layers:

 the path-0 layers is
    Conv2DTranspose
    Concatenate
    Conv2D
    BatchNormalization
    Activation
    Conv2D
    BatchNormalization
    Activation
    ...
    Conv2D (final 1x1)
 the path-1 layers is
    ...
 the path-2 layers is
    ...
```

---

## 🧪 All Parameters

| Parameter          | Type          | Description                                               |
| ------------------ | ------------- | --------------------------------------------------------- |
| `filters`          | list of int   | Filters used in encoder (reversed for decoder)            |
| `kernel_size`      | tuple of ints | Must be odd numbers, e.g. (3,3), (5,5)                    |
| `batch_norm`       | bool          | Whether to apply BatchNormalization                       |
| `activation`       | str or None   | Hidden layer activation (e.g., 'relu', 'tanh', 'sigmoid') |
| `final_activation` | str or None   | Output activation for each decoder path                   |
| `out_channels`     | int           | Number of output decoder branches (multi-output)          |

---

## 📌 Notes

* Each decoder path operates independently, sharing the same encoder and skip connections (per U-Net design).
* Final output is concatenation of all decoder path results along the channel axis.
* Useful for multi-field regression/segmentation (e.g., CFD with multiple quantities).
* This architecture omits weight normalization, using standard BatchNormalization.

---

## ✅ License

MIT License
