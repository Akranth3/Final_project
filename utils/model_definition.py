from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model


def build_densenet_based_model():
    base_model = DenseNet121(include_top=False, input_shape=(256, 256, 3), weights='imagenet')
    x = base_model.output

    # Adding upsampling layers
    x = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)  # size becomes 16x16
    x = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)  # size becomes 32x32
    x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)  # size becomes 64x64
    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)  # size becomes 128x128
    x = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)  # size becomes 256x256

    # Final output layer
    outputs = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)  # Ensure single channel output

    model = Model(inputs=base_model.input, outputs=outputs)
    return model