from torchvision import transforms

preprocess = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(30),
        transforms.CenterCrop(28),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.456], std=[0.224]),
    ]
)