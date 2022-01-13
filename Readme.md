**Cloning the repository**	

`git clone https://github.com/krishansinghal29/imageclassification_pubsub.git`

**Setting up python environment**

`python -m venv pyenv1`

**Shifting to python environment** 

`source pyenv1/bin/activate`

**Installing the required dependencies** 

`pip install requirements.txt`

**The following functions are supported by the library. All the following commands should be executed from root of repository.**

1. Training the model
   
   i) Training the model on mnist dataset
   
   `python3 main.py train`
   
   ii) Optionally provide the "False" argument to train the model on custom dataset
   
   `python3 main.py train False`
   
   If training on a custom dataset, the train data should be stored in the data/train_data folder.
   
   data/train_data should contain subfolders containing images for each class.
2. Testing the model
   
   i) Testing the model on mnist dataset:
   
   `python3 main.py test`
   
   ii) Optionally provide the "False" argument to test the model on custom dataset
   
   `python3 main.py test False`
   
   If testing on a custom dataset, the test data should be stored in the data/test_data folder.
   data/test_data should contain subfolders containing images for each class.
3. Performing prediction on single image
   
   `python3 predict.py -path_to_imagefile`
4. Sending the image file to model server for classification
   
   `python3 publish.py -path_to_imagefile`
5. Starting the model server and run classification on received requests
   
   `python3 subscribe.py`
