import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
# ------------------- Separete formats and delete teh format jfif ----------------------
import os

num_skipped = 0
for folder_name in ():
    folder_path = os.path.join(r"C:\Users\Manuel\Desktop\Imagenes_30Metros\TITI\CIMA", folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)

print("Deleted %d images" % num_skipped)

#  ---------------------- prepare dataset -------------------
# Model / data parameters
img_size = (180,180)
batch_size = 10

x_train, y_train = tf.keras.utils.image_dataset_from_directory(
    "TiTi.v2i.yolov5pytorch (1)",
    validation_split=0.2,
    subset= "both",
    seed=1337,
    image_size=img_size,
    batch_size=batch_size,
)
# ------------------------------ Data augmentation--------------
data_aug = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
)

aug_x_train = x_train.map(
    lambda x, y: (data_aug(x, training=True), y)
)

# ------------------------------ dataset performance -------------
x_train = x_train.map(
    lambda img, label: (data_aug(img), label),
    num_parallel_calls = tf.data.AUTOTUNE,
)

x_train = x_train.prefetch(tf.data.AUTOTUNE)
y_train = y_train.prefetch(tf.data.AUTOTUNE)


# ------------------------- Building Model ------------------------
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    x = layers.Rescaling(1/255)(inputs)
    x = layers.Conv2D(128,3,strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    prev = x

    for size in [256,512,728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size,3,padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size,3,padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        res = layers.Conv2D(size, 1, strides=2, padding="same")(
            prev
        )
        x = layers.add([x, res])
        prev = x

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        acti = "sigmoid"
        units = 1
    else:
        acti = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    out = layers.Dense(units, activation=acti)(x)
    return keras.Model(inputs, out)

model = make_model(input_shape=img_size + (3,), num_classes=2)
keras.utils.plot_model(model, show_shapes=True)

epochs = 30

callb = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss = tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy()],
)

model.fit(
    x_train,
    epochs=epochs,
    callbacks=callb,
    validation_data=y_train,
)

# ----------------------------------------- If we wanna try it ------------------------------------------------

#img = keras.utils.load_img(
#    "PetImages/Cat/6779.jpg", target_size=image_size
#)
#img_array = keras.utils.img_to_array(img)
#img_array = tf.expand_dims(img_array, 0)  # Create batch axis

#predictions = model.predict(img_array)
#score = float(predictions[0])
#print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")