from torch import max, mean, FloatTensor, save, no_grad, load
from torchvision.models import vgg11, VGG11_Weights
from torch.nn import Linear
from torch.utils.data import DataLoader


def computeTestAccuracy(model: vgg11, testLoader: DataLoader) -> float:
    """Return the test accuracy of the given model on the given test set.

    Args:
        model (vgg11): model.
        testLoader (DataLoader): test set.

    Returns:
        float: test accuracy.
    """
    with no_grad():
        model.eval()
        test_acc = 0
        for data, label in testLoader:
            data, label = data.cuda(), label.cuda()

            output = model(data)

            pred = max(output, dim=1)[1]
            correct_tensor = pred.eq(label.data.view_as(pred))
            accuracy = mean(correct_tensor.type(FloatTensor))
            test_acc += accuracy.item() * data.size(0)

        test_acc = test_acc / len(testLoader.dataset)
        return test_acc


def load_model_from_checkpoint(load_accuracy: bool = False) -> tuple[vgg11, float]:
    """Load from the memory and return a stored model along along with its accuracy.
    If no model is found, then a tuple of Nones is returned.

    Args:
        path (str): path where the model is stored.
        load_accuracy (bool, optional): If True also the accuracy of the
        said model is returned. Defaults to False.

    Returns:
        tuple[vgg11, float]: _description_
    """
    model = vgg11(weights=VGG11_Weights.DEFAULT)
    numberOfClasses = 2
    model.classifier[6] = Linear(4096, numberOfClasses)
    testAccuracy = 0
    try:
        torch_checkpoint = load("classifier.pt")
        model.load_state_dict(torch_checkpoint["model_state_dict"])
        model.idx_to_class = torch_checkpoint["idx_to_class"]
        if load_accuracy:
            testAccuracy = torch_checkpoint["test_acc"]
        model.cuda()
    except:
        return None, None
    return model, testAccuracy


def storeModelVGG1(vgg11Model: vgg11, testAccuracy: float) -> None:
    """Save the given model as checkpoint along with its test accuracy
    and mapping idx-classes.

    Args:
        vgg11Model (vgg11): vgg11 model to store.
        testAccuracy (float): test accuracy of the given model.
    """
    # Kept different naming convention to respect pytorch checkpoint naming convetion.
    save(
        {
            "model_state_dict": vgg11Model.state_dict(),
            "idx_to_class": vgg11Model.idx_to_class,
            "test_acc": testAccuracy,
        },
        "classifier.pt",
    )


def compareModels(model: vgg11, testLoader: DataLoader) -> None:
    """Compare the given model along with the best stored in the memory through their
    test accuracies.
    If no model is found, then the current model is stored as best model if its test
    accuracy is higher than 80%.

    Args:
        model (vgg11): new model.
        testLoader (DataLoader): test set/loader of the current model.
    """
    loadedModel, loadedModelTestAccuracy = load_model_from_checkpoint(
        load_accuracy=True
    )
    testAccuracy = computeTestAccuracy(model, testLoader) * 100.0
    comparisonTrace = f"Current model test accuracy: {testAccuracy}.\n"
    if loadedModel == None:
        comparisonTrace += "No model stored found.\n"
        storeModelVGG1(vgg11Model=model, testAccuracy=testAccuracy)
        comparisonTrace += (
            f"New model's test accuracy: {testAccuracy}. New model saved.\n"
        )
    else:
        if testAccuracy > loadedModelTestAccuracy:
            storeModelVGG1(vgg11Model=model, testAccuracy=testAccuracy)
            comparisonTrace += f"The new model has a HIGHER accuracy than the one stored.\nNew model saved.\n"
        else:
            comparisonTrace += f"The new model has a LOWER accuracy than the one stored:\n - STORED: {loadedModelTestAccuracy}\n - NEW: {testAccuracy}\n"
    print(comparisonTrace)
