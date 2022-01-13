**Cloning the repository**

`git clone https://github.com/krishansinghal29/imageclassification_pubsub.git`

**Move to the repository**

`cd imageclassification_pubsub`

**Setting up python environment. Install python3 before running this command if not already installed**

`python3 -m venv pyenv1`

**Shifting to python environment**

`source pyenv1/bin/activate`

**Installing the required dependencies**

`pip install -r requirements.txt`

**The following functions are supported by the library:**

All the following commands should be executed from root of repository

1. Training the model

   i) Training the model on mnist dataset

   `python3 src/main.py train`

   ii) Optionally provide the "False" argument to train the model on custom dataset

   `python3 src/main.py train False`

   If training on a custom dataset, the train data should be stored in the data/train_data folder.

   data/train_data should contain subfolders, each subfolder containing images for each label class.

2. Testing the model

   i) Testing the model on mnist dataset:

   `python3 src/main.py test`

   ii) Optionally provide the "False" argument to test the model on custom dataset

   `python3 src/main.py test False`

   If testing on a custom dataset, the test data should be stored in the data/test_data folder.
   data/test_data should contain subfolders, each subfolder containing images for each label class.

3. Performing prediction on single image

   `python3 src/predict.py -path_to_imagefile`

4. Sending the image file to model server for classification

   `python3 src/publish.py -path_to_imagefile`

5. Starting the model server and run classification on received requests

   `python3 src/subscribe.py`
