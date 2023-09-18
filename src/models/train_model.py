import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam


def train_model(model_name, train_generator, validation_generator, base_model=None):
    if base_model is None:
        return train_custom_model(model_name, train_generator, validation_generator)
    if base_model == 'ResNet':
        return train_resnet_model(model_name, train_generator, validation_generator)
    if base_model == 'MobileNet':
        return train_mobilenet_model(model_name, train_generator, validation_generator)


def train_custom_model(model_name, train_generator, validation_generator, image_height=224, image_width=224):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),  # Dropout layer to reduce overfitting
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=1e-4),
                  metrics=['accuracy', tfa.metrics.F1Score(num_classes=1, average='micro', threshold=0.5)]
    )

    history = model.fit(train_generator, epochs=100, validation_data=validation_generator)
    model.save(model_name)
    return model


def train_resnet_model(model_name, train_generator, validation_generator, image_height=224, image_width=224):
    height, width = 224, 224
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False,
                                                input_shape=(image_height, image_width, 3))

    # Freeze the base model's layers
    base_model.trainable = False

    # Build the new model using the base model and adding new top layers
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model with F1 score as a metric
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        #     optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy', tfa.metrics.F1Score(num_classes=1, average='micro', threshold=0.5)]
    )

    # Train the model for a fixed number of epochs
    history = model.fit(
        train_generator,
        epochs=30,  # Adjust the number of epochs
        validation_data=validation_generator,
    )

    model.save(model_name)
    return model


def train_mobilenet_model(model_name, train_generator, validation_generator, image_height=224, image_width=224):
    height, width = 224, 224
    # Load the pre-trained MobileNet model without the top layers
    base_model = tf.keras.applications.MobileNet(weights='imagenet', include_top=False,
                                                 input_shape=(image_height, image_width, 3))

    # Freeze the base model's layers
    base_model.trainable = False

    # Build the new model using the base model and adding new top layers
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model with F1 score as a metric
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        metrics=['accuracy', tfa.metrics.F1Score(num_classes=1, average='micro', threshold=0.5)]
    )

    # Train the model for a fixed number of epochs
    history = model.fit(
        train_generator,
        epochs=30,  # Adjust the number of epochs
        validation_data=validation_generator,
    )

    model.save(model_name)
    return model
