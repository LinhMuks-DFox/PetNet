import enum
import torch.utils.data as tch_data
from torchvision.datasets import ImageFolder
import torchvision.transforms as visionT

train_transform = visionT.Compose([
    visionT.RandomResizedCrop(224),
    visionT.RandomHorizontalFlip(),
    visionT.ToTensor(),
    visionT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
eval_transform = visionT.Compose([
    visionT.Resize(256),
    visionT.CenterCrop(224),
    visionT.ToTensor(),
    visionT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class DatasetType(enum.Enum):
    Train = "Train"
    Eval = "Eval"
    Test = "Test"

    dataset = ImageFolder("./PetImages/")


train_set, validate_set, test_set = tch_data.random_split(dataset, [
    0.8, 0.1, 0.1])
train_set.dataset.transform = train_transform
validate_set.dataset.transform = eval_transform
test_set.dataset.transform = eval_transform


def get_dataset(type: DatasetType):
    if type == DatasetType.Train:
        return train_set
    elif type == DatasetType.Eval:
        return validate_set
    elif type == DatasetType.Test:
        return test_set
    else:
        raise ValueError("Invalid dataset type")
