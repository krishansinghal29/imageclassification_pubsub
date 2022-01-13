import sys, os
sys.path.append(os.path.dirname(__file__))

import model_cnn
import preprocess_module 
preprocess=preprocess_module.preprocess

import torch
from PIL import Image
import argparse
import base64
from io import BytesIO

def predict(data_path):
    model = model_cnn.Net()
    model.load_state_dict(torch.load("./model_dir/trained_model.pt"))
    # Test the model
    model.eval()

    img = Image.open(data_path)

    img_tensor = preprocess(img).unsqueeze(0)
    output = model(img_tensor)
    pred_y = torch.max(output, 1)[1].data.squeeze().item()
    print("Predicted Label for file is: ",pred_y)


def predict_decode(data):
    model = model_cnn.Net()
    model.load_state_dict(torch.load("./model_dir/trained_model.pt"))
    # Test the model
    model.eval()

    img = Image.open(BytesIO(base64.b64decode(data)))

    img_tensor = preprocess(img).unsqueeze(0)
    output = model(img_tensor)
    pred_y = torch.max(output, 1)[1].data.squeeze().item()
    print("Predicted Label for file is: ",pred_y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify the image using trained model")
    parser.add_argument(
        "file_path", help="file_path"
    )
    args = parser.parse_args()

    predict(args.file_path)

    # with open(args.file_path, "rb") as image_file:
    #     data = base64.b64encode(image_file.read())
    # print(data)
    # predict_decode(data)