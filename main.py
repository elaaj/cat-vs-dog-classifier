# Initialization utilities.
from trainingInitialization import (
    cudaInitialization,
    initImageTransforms,
    retrieveData,
    initDataLoaders,
    initModelVGG11,
    activateCuda,
)
from torch.nn import CrossEntropyLoss

# Training utilities.
from vggTrainer import train
from torch.optim import AdamW

# Evaluation utilities.
from trainingEvaluation import compareModels


# TO RUN use the environment ai_cv_test as follows:
#   *   mamba activate ai_cv_test
# And run the program:
#   *   python main.py


def main():
    #############################################################################################################
    ###############################################               ###############################################
    #############################################   INTIALIZATION   #############################################
    ###############################################               ###############################################
    #############################################################################################################

    # Retrieve the reference to the cuda module to afflict the current program
    # and also the running device for a dynamic execution, alowing for CPU/GPU runs.
    cudaModuleRef, runningDevice = cudaInitialization()
    print(runningDevice)
    

    # Retrieve the set of online image augmentations required
    # during the training.
    imageTransforms = initImageTransforms()

    # Retrieve the data
    data = retrieveData()

    # Retrieve the initialized data loaders.
    trainingLoader, validationLoader, testLoader = initDataLoaders(
        data=data, imageTransforms=imageTransforms, batchSize=62, numberOfCPUWorkers=3
    )

    # trainingLoader.to(runningDevice)

    # Retrieve a VGG11 model with pretrained weights
    model = initModelVGG11(data=data)
    # print(model)
    # print(model.classifier)

    # Check cuda and load the model on the GPU
    if not activateCuda(cudaModuleRef=cudaModuleRef, model=model):
        raise Exception("Cuda is not available for training!")

    # Transfer-L VGG has the softmax output layer removed, so we  need to keep
    # it on account into the loss function.
    lossFunction = CrossEntropyLoss()

    #############################################################################################################
    ###############################################               ###############################################
    #############################################     TRAINING      #############################################
    ###############################################               ###############################################
    #############################################################################################################

    print("\n\n---STARTING TRAINING---\n")

    model = train(
        model=model,
        lossFunction=lossFunction,
        # lr=1e-3 is used just as placeholder
        optimizer=AdamW(params=model.parameters(), lr=1e-3, weight_decay=1e-4),
        trainingDataLoader=trainingLoader,
        validationDataLoader=validationLoader,
        learningRates=[0.00008, 0.000005, 0.00000005],
    )

    print("---TRAINING COMPLETED---\n")


    #############################################################################################################
    ###############################################               ###############################################
    ##############################################    EVALUATION   ##############################################
    ###############################################               ###############################################
    #############################################################################################################


    # Compare the model just trained with the stored one.
    compareModels(
        model,
        testLoader,
    )


if __name__ == "__main__":
    main()
