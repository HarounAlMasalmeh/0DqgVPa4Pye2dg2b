
def predict_model(model, test_generator):
    test_loss, test_accuracy, test_f1 = model.evaluate(test_generator)
    return test_loss, test_accuracy, test_f1
