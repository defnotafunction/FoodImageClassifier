from PIL import Image
import torch
import torch.nn as nn
import os
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(device)

FAST_FOOD_TRAINING_PATH = os.path.join('FastFoodImages', 'Train')
FAST_FOOD_TESTING_PATH = os.path.join('FastFoodImages', 'Test')
FOOD_NAMES = [
    'Baked Potato', 'Burger', 'Crispy Chicken', 'Donut',
    'Fries', 'Hot Dog', 'Pizza', 'Sandwich', 'Taco', 'Taquito'
          ]
FOOD_TO_IDX = {
    name: idx for idx, name in enumerate(FOOD_NAMES)
}

IDX_TO_FOOD = {
    idx: name for idx, name in enumerate(FOOD_NAMES)
}

NUMBER_OF_LABELS = len(FOOD_NAMES)
IMAGE_SIZE = 224

# DATA AUGEMENTATION
train_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# REGULAR IMAGE TRANSFORM
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
        )
])

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # CONVOLUTION LAYERS
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # CLASSIFIER
        self.fc_layers = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, NUMBER_OF_LABELS)
        )

        # ACTIVATION FUNCTION
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)

        return x

def convert_image_file_to_tensor(file_path: str, train_mode=False) -> torch.Tensor:
    """
    Opens image file, returns a transformed tensor form.
    
    :param file_path: File path of image.
    :param train_mode: Boolean if True applies a training version of a transformation otherwise applies the regular transformation.
    :return: Image tensor of shape (3, 224, 224)
    """

    img = Image.open(file_path)

    if train_mode:
        img_tensor = train_transform(img)
    else:
        img_tensor = transform(img)

    return img_tensor

def get_fast_food_image_data(path: str, train_mode=False) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Iterates through image file in folder, transforms each image, returns Tensor of images with respective labels.
    
    :param file_path: File path of image.
    :param train_mode: If true applies data augmentation transformations, otherwise applies standard transformations.
    :return: 2D tensor of examples/transformed images, 1D tensor of labels.
    """

    examples, labels = [], []

    for name in FOOD_NAMES:
        food_path = os.path.join(path, name)
        files = os.listdir(food_path)

        for file_name in files:
            file_path = os.path.join(food_path, file_name)
            img_tensor = convert_image_file_to_tensor(file_path, train_mode=train_mode)

            examples.append(img_tensor)
            labels.append(FOOD_TO_IDX.get(name))

    return torch.stack(examples), torch.tensor(labels)

def train_model(model: nn.Module, num_epochs: int) -> None:
    """
    Trains given model for num_epochs epochs.
    
    :param model: A neural network of the torch.nn.Module class.
    :param num_epochs: Number of epochs.
    """

    train_examples, train_labels = get_fast_food_image_data(FAST_FOOD_TRAINING_PATH, train_mode=True)
    dataset = TensorDataset(train_examples, train_labels)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train() # Training mode.
    
    for i in range(num_epochs):
        # TRAINING LOOP
        for batch_idx, (data, target) in enumerate(train_loader):
            # MOVE DATA TO GPU FOR ACCELERATION
            data, target = data.to(device), target.to(device)
            
            predictions = model(data)

            loss = criterion(predictions, target)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            if i % 100 == 0 and batch_idx == 0:
                preds = torch.argmax(predictions, dim=1)
                acc = (preds == target).float().mean()
                torch.save(model.state_dict(), 'model.pth') 
                print(f'Epoch {i+1}: Loss - {loss.item()} | Accuracy - {acc.item()} | Batch - {batch_idx}')

def get_model_metrics(model: nn.Module) -> None:
    """
    Prints out accuracy of model.
    
    :param model: A neural network of the torch.nn.Module class.
    """

    model.eval() # Evaluation mode.

    with torch.no_grad():
        X_test, y_test = get_fast_food_image_data(FAST_FOOD_TESTING_PATH, train_mode=False)
        X_test = X_test.to(device)
        y_test = y_test.to(device)

        output = model(X_test)
        probabilities = torch.softmax(output, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

        print(f'ACCURACY: {accuracy_score(y_test.cpu().numpy(), predictions.cpu().numpy())}')



cnn = CNN().to(device)
cnn.load_state_dict(torch.load('model.pth', weights_only=True))

train_model(cnn, 100)

#torch.save(cnn.state_dict(), 'model.pth')

cnn.eval()
burger_image = os.path.join('burger.jpg')
img = convert_image_file_to_tensor(burger_image)

img = img.to(device)
img = img.unsqueeze(0)

with torch.no_grad():
    output = cnn(img)
    probabilities = torch.softmax(output, dim=1)
    prediction = torch.argmax(probabilities, dim=1).item()
    print(f'This is a {IDX_TO_FOOD.get(prediction)}')

get_model_metrics(cnn)