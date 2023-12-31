import argparse

from keras.metrics import Accuracy
from tensorflow.keras.models import load_model
import tensorflow_addons as tfa

from src.features.build_features import build_features
from src.models.predict_model import predict_model
from src.models.train_model import train_custom_model

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A simple command line arguments parser")

    parser.add_argument("-d", "--data", type=str, default="../data/raw/images/", help="data folder path")
    parser.add_argument("-m", "--model", type=str, default="../notebooks/MobileNet_MonReader.h5",
                        help="Pre-trained model path")

    args = parser.parse_args()

    print(args)

    data_file_path = args.data
    model_name = args.model

    training_data_path, testing_data_path = data_file_path + "/training", data_file_path + "/testing"

    train_generator, validation_generator, test_generator = build_features(training_data_path, testing_data_path)

    # model = train_model(model_name, train_generator, validation_generator)

    custom_objects = {'Accuracy': Accuracy,
                      'F1Score': tfa.metrics.F1Score(num_classes=1, average='micro', threshold=0.5)}
    model = load_model(model_name, custom_objects=custom_objects)
    test_loss, test_accuracy, test_f1 = predict_model(model, test_generator)
    print(f'Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, F1_Score: {test_f1:.4f}')
