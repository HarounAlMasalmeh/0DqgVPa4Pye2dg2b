import tensorflow_addons as tfa
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam


def train_model(model_name, train_generator, validation_generator):
    height, width = 200, 100
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, 3)),
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
                  metrics=[tfa.metrics.F1Score(num_classes=1, average='micro', threshold=0.5)]
    )

    history = model.fit(train_generator, epochs=100, validation_data=validation_generator)
    model.save(model_name)
    return model
