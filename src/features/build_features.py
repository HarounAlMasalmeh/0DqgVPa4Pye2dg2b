from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_features(training_data_path, testing_data_path):
    height, width = 200, 100
    size = (height, width)

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    # training_data_path = "../data/raw/images/training"
    # testing_data_path = "../data/raw/images/testing"

    train_generator = train_datagen.flow_from_directory(training_data_path, target_size=size,batch_size=32,
                                                        class_mode='binary', subset='training')

    validation_generator = train_datagen.flow_from_directory(training_data_path, target_size=size,
                                                             batch_size=32, class_mode='binary',subset='validation')

    test_generator = test_datagen.flow_from_directory(testing_data_path, shuffle=True, target_size=size, batch_size=32,
                                                      class_mode='binary')

    return train_generator, validation_generator, test_generator
