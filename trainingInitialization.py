from albumentations import (
    Compose,
    Normalize,
    CenterCrop,
    HueSaturationValue,
    RandomBrightnessContrast,
    Emboss,
    Sharpen,
    CLAHE,
    PiecewiseAffine,
    GridDistortion,
    OpticalDistortion,
    ShiftScaleRotate,
    Blur,
    MedianBlur,
    MotionBlur,
    OneOf,
    GaussNoise,
    Transpose,
    Flip,
    RandomRotate90,
)
from albumentations.pytorch import ToTensorV2
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split
from torch import (
    cuda,
    device,
    backends,
    autograd,
)
from torch.nn import Linear


def cudaInitialization():
    """Initialize the cuda module for the training, retrieving also the available
    device (cuda or CPU).

    Returns:
        tuple[cuda, str]: tuple containing the module and the string.
    """
    # Empty the GPU memory (in case of unallocated elements in it).
    cuda.init()
    cuda.empty_cache()
    cuda.memory_summary(device=None, abbreviated=False)

    # Deactivate profiling for lower memory/cpu usage.
    autograd.profiler.profile(enabled=False)
    autograd.profiler.emit_nvtx(enabled=False)
    autograd.set_detect_anomaly(mode=False)

    # Return the cuda module and the current device.
    return cuda, device("cuda:0" if cuda.is_available() else "cpu")


def initImageTransforms():
    """Return a Composition of all the required augmentations
    for training and test.
    Training uses real augmentations, whereas Test just uses the
    transformations needed to format the images as required by 
    ImageNet standard.

    Returns:
        dict: Dictionary {train: augmentations, test: formatting}
    """
    return {
        # Train uses data augmentation
        "train": Compose(
            [
                RandomRotate90(),
                Flip(),
                Transpose(),
                GaussNoise(p=0.2),
                OneOf(
                    [
                        MotionBlur(p=0.2),
                        MedianBlur(blur_limit=3, p=0.1),
                        Blur(blur_limit=3, p=0.1),
                    ],
                    p=0.2,
                ),
                ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2
                ),
                OneOf(
                    [
                        OpticalDistortion(p=0.3),
                        GridDistortion(p=0.1),
                        PiecewiseAffine(p=0.3),
                    ],
                    p=0.2,
                ),
                OneOf(
                    [
                        CLAHE(clip_limit=2),
                        Sharpen(),
                        Emboss(),
                        RandomBrightnessContrast(),
                    ],
                    p=0.3,
                ),
                HueSaturationValue(p=0.3),
                CenterCrop(
                    height=299, width=299, always_apply=True
                ),  # Image net standards
                Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),  # Imagenet standards
                ToTensorV2(),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(size=299),
                transforms.CenterCrop(size=299),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }


def retrieveData(dataPath: str = "DATASET/train/") -> datasets.ImageFolder:
    """Return the data through the ImageFolder pytorch utility.

    Args:
        dataPath (str, optional): path to the folder containing the images.
        Defaults to "DATASET/train/".

    Returns:
        datasets.ImageFolder: wrapper of images.
    """
    return datasets.ImageFolder(root=dataPath)


def initDataLoaders(
    data: datasets.ImageFolder,
    imageTransforms: dict,
    batchSize: int = 16,
    numberOfCPUWorkers=0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Splits the data into training, testing and validation loaders for training and testing
    purposes.

    Args:
        imageTransforms (dict): transforms to be applied to the data.
        batchSize (int): batch size chosen for the training. Defaults to 62.
        numberOfCPUWorkers (int): number of worker threads/processes which will
        handle the loading of the data from the memory. Defaults to 4.

        ATTENTION: workers number and batch size can afflict heavily the execution
        of the program (even causing failures).

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: intialized dataloaders.
    """

    dataSize = len(data)
    trainingDataSize = int(dataSize * 0.8)
    validationDataSize = int((dataSize - trainingDataSize) / 2)
    testDataSize = int(dataSize - trainingDataSize - validationDataSize)
    trainingData, validationData, testData = random_split(
        data, [trainingDataSize, validationDataSize, testDataSize]
    )
    trainingData.dataset.transform = imageTransforms["train"]
    validationData.dataset.transform = imageTransforms["test"]
    testData.dataset.transform = imageTransforms["test"]
    # return Train, Val, and Test loaders.
    return (
        DataLoader(
            trainingData,
            num_workers=numberOfCPUWorkers,
            pin_memory=True,
            batch_size=batchSize,
            shuffle=True,
        ),
        DataLoader(
            validationData,
            num_workers=numberOfCPUWorkers,
            pin_memory=True,
            batch_size=batchSize,
            shuffle=True,
        ),
        DataLoader(
            testData,
            num_workers=numberOfCPUWorkers,
            pin_memory=True,
            batch_size=batchSize,
            shuffle=True,
        ),
    )


def initModelVGG11(data: datasets.ImageFolder):
    """Build and return a VGG 11 model with pre-trained weights.
    The last layer has been readapted to 2 outputs, as required for
    a binary classification, all the layers of the model have been
    activated for training, and the mapping class-index is stored in
    the model. 

    Args:
        data (datasets.ImageFolder): pytorch utility used to read data.

    Returns:
        _type_: _description_
    """
    # Load basic model with pre-trained weights.
    model = models.vgg11(weights=models.VGG11_Weights.DEFAULT)

    # Train every layer, even feature extractor.
    for param in model.parameters():
        param.requires_grad = True

    # Replace the last Linear layer with an identical layer with just 2 outputs
    # and not 1000.
    model.classifier[6] = Linear(in_features=4096, out_features=2)

    # Initalize classes' indexes for class discrimination.
    model.class_to_idx = data.class_to_idx
    model.idx_to_class = {idx: class_ for class_, idx in model.class_to_idx.items()}

    return model


def activateCuda(cudaModuleRef: cuda, model: models.vgg11) -> bool:
    """Check whether cuda is available and load the model on the GPU.

    Args:
        cudaModuleRef (cuda): cuda module reference.
        model (models.vgg11): CNN model.

    Returns:
        bool: True if cuda is available, False otherwise.
    """
    if cudaModuleRef.is_available():
        model.cuda()
        backends.cudnn.benchmark = True
        return True
    return False
