from torch import cuda, max, mean, FloatTensor, no_grad
from torchvision.models import vgg11


def train(
    model,
    lossFunction,
    optimizer,
    trainingDataLoader,
    validationDataLoader,
    learningRates,
) -> vgg11:
    """Train the model using the given optimizer and estimating the 
    errors and accuracies on the validation set and training set
    using the given loss function.
    The model will be trained using for each epoch one of the learning rates
    given, until no more will be available.

    Args:
        model (vgg11): vgg11 model to train
        lossFunction (CrossEntropy): cross entropy loss function.
        optimizer (AdamW): weighted adam optimizer.
        trainingDataLoader (DataLoader): training data loader.
        validationDataLoader (DataLoader): validation data loader.
        learningRates (list[float]): list of learning rates.

    Returns:
        vgg11: trained model.
    """
    trainingDataLoaderLength = len(trainingDataLoader.dataset)
    validationDataLoaderLength = len(validationDataLoader.dataset)
    trainingLenForPrint = len(trainingDataLoader)

    # Epochs: there are as much as given learning rates.
    for epochsIndex, epochsLearningRate in enumerate(learningRates):

        # Initialize accuracies and losses to 0.
        train_loss, valid_loss, train_acc, valid_acc = 0, 0, 0, 0

        # Set the learning rate of the current epoch on every parameter
        # of the model.
        for param_group in optimizer.param_groups:
            param_group["lr"] = epochsLearningRate

        # Store the Learning rate of the current parameter.
        actual_lr = epochsLearningRate

        # Set the model in training mode
        model.train()

        # Keep count of the loaded batches: used to print the current 
        # epoch progress.
        loadedBatches = 0

        # Cuda AMP: allows for an automatic conversion of variables to FloatingPoint 16-bit
        # instead of 32 when possible.
        scaler = cuda.amp.GradScaler()

        # Start to iterate through the Training Batches
        for data, label in trainingDataLoader:
            loadedBatches += 1

            # Load the current batch on the GPU
            data, label = data.cuda(), label.cuda()

            optimizer.zero_grad(set_to_none=True)

            with cuda.amp.autocast():
                # Compute the output and the loss over the current batch.
                output = model(data)
                loss = lossFunction(output, label)

            # Update the AMP scaler and the Adam Optimizer.
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Compute training loss: multiply average loss by number of examples in batch
            train_loss += loss.item() * data.size(0)

            # Calculate accuracy by finding max log probability
            _, pred = max(output, dim=1)
            train_acc += mean(
                pred.eq(label.data.view_as(pred)).type(FloatTensor)
            ).item() * data.size(0)

            # Every 10 batches print the current epoch progress.
            if loadedBatches % 10 == 0:
                print(
                    f"Epoch: {epochsIndex}\t{100 * (loadedBatches + 1) / trainingLenForPrint:.2f}% complete."
                )
        
        # Validation
        with no_grad():
            model.eval()

            # Iterate through validation batches.....
            for data, label in validationDataLoader:

                data, label = data.cuda(), label.cuda()

                with cuda.amp.autocast():
                    output = model(data)
                    loss = lossFunction(output, label)
                valid_loss += loss.item() * data.size(0)

                _, pred = max(output, dim=1)
                valid_acc += mean(
                    pred.eq(label.data.view_as(pred)).type(FloatTensor)
                ).item() * data.size(0)

            train_loss /= trainingDataLoaderLength
            valid_loss /= validationDataLoaderLength

            train_acc /= trainingDataLoaderLength
            valid_acc /= validationDataLoaderLength

            print(
                (
                    f"EPOCH N°: {epochsIndex} \n   Training Loss: {train_loss:.4f} \n   Validation Loss: {valid_loss:.4f}"
                    f" \n   Training Accuracy: {(100 * train_acc):.2f}% \n   Validation Accuracy: {(100 * valid_acc):.2f}"
                    f"% \n   Learning Rate: {actual_lr:.9f}"
                )
            )

    model.optimizer = optimizer
    print(
        f"\nAchieved validation loss: {valid_loss:.2f} and accuracy: {100 * valid_acc:.2f}%"
    )

    return model
