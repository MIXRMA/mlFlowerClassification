# Introduction: Flower Classification Using ML
This project is a flower species classification tool that is trained on the Oxford 102 Flower dataset (~8,200 images across 102 classes), downloaded directly from the VGG servers at Oxford without using Kaggle or GitHub. This project has been built iteratively, beginning with a dataset download script (```downloadData.py```) that fetches and organizes the raw images ito per-class folders, followed by a ```config.py```for centralized hyperparameter and path management, then a full CNN architecture, training loop, evaluator and the Flask inference interface. The final CNN achieved 58% Top-1 and 83% Top-5 accuracy on 102 classes with no pretrained weights, which is a strong result for a from-scratch model. The Flaks interface accepts images of any format, size, colour profile or lighting condition, including images fetched from URLs, and returns the predicted flower class with a confidence percentage.

# Python Libraries
1. ```torch```: This library is the core framework for defining the custom CNN architecture, managing the training loop, computing cross-entropy loss and running backpropagation with MPS acceleration.
2. ```torchvision```: This library provides ```transforms```for resizing, normalizing and tensor-converting Oxford 102 images during both training and inference.
3. ```numpy```: This library handles numerical array operations during dataset loading, label parsing and preprocessing steps before images are converted to PyTorch tensors.
4. ```Pillow```: This library opens and decodes JPEG images from the Oxford 102 dataset and handles user-uploaded images in the Flask interface regardless of format, colour profile or any other parameter.
5. ```scipy```: This library parses the ```.mat``` files (```ìmagelabels.mat``` and ```setid.mat```) that accompany the Oxford 102 dataset and encode the ground-truth class labels and official train/val/test splits.
6. ```matplotlib```: This library plots the training and validation loss/accuracy curves across epochs and generates per-class performance visualizations after evaluation.
7. ```tqdm```: This library wraps dataset download loops and training epoch iterations with real-time progress bars in the terminal.
8. ```Flask```: This library serves the web interface that accepts image iploads or URL-based inputs and returns the predicted flower class with confidence percentage from the trained CNN.
9. ```Werkzeug```: This library handles secure file upload operations within Flask, santizing filenames and managing temporary file storage for uploaded images.
10. ```requests```: This library fetches images from arbitrary internet URLs (For example: Google Images links) submitted through the Flask interface so inference can run on web-sourced images.
11. ```urllib3```: This library underlies the ```requests```library and manages the low-level HTTP connection pooling used when fetching remote images.
12. ```colorama```: This library adds coloured terminal output diring training and dataset download steps on macOS, thus making status messages and error logs visually distinguishable.

# Training Model
In this project, a custom CNN model is used as the training model. It is built from scratch in PyTorch (Without any pretrained weights and transfer learning), trained with MPS acceleration on the M4 chip (I use an M4 MacBook Air). The architecture uses progressive filter doubling across convolutional blocks with BatchNorm, Dropout and MaxPooling, terminating in a fully connected classifier head with Softmax output over 102 classes.

# Project Directory (Textual Format)
<img width="229" height="682" alt="image" src="https://github.com/user-attachments/assets/6fcf08e6-aece-46db-8ae3-c84c7e61e7b9"/>

# Instructions To Execute
1. Add all the files and folders exactly in the same order as the image of the project directory that I have attached above (Except the ```raw/``` folder under ```data/```, leave just that one empty; Explained in Point 3). Once that is taken care of, add an empty ```__init__.py``` file in the ```augment/```, ```models/```, ```train/``` and ```evaluate/``` folders.
2. Set up the virtual environment (```venv```) for this project in the Terminal and install the libraries by running the command ```pip install -r requirements.txt```.
3. Run the command ```python main.py --mode download```. On running this, ```downloadData.py``` will fetch three files: ```102flowers.tgz``` (~330 MB; The raw images), ```imagelabels.mat``` (The MATLAB label array; Gets saved to ```data/raw/``` mentioned earlier) and ```setid.mat``` (The official train, val and test split indices; Also saved to ```data/raw/```). The ```.tgz``` file is extracted to ```data/raw/oxford102/jpg/``` (Refer the project directory above), producing a flat directory of 8,189 sequentially named JPEG files with no class structure.
4. During the download phase, ```downloadData.py``` reads ```imagelabels.mat``` and iterates over all 8,189 images and sorts each one into a named class subfolder under the ```organised/``` folder in ```data/raw/oxford102/```. The official ```setid.mat``` of Oxford 102 allocates only 10 training images per class (Bringing the total training images to 1,020 as 10 * 102 = 1,020), which is very less for a fresh CNN model to learn from. Therefore, official splits are ignored and stratified random split across the whole dataset of 8.189 images is performed randomly in a 70/15/15 split (test, train and val respectively), and configured finally in ```config.py```.
5. Run the command ```python main.py --mode train``` in the Terminal to train the CNN model on the dataset, which on an average takes around 20-30 minutes depending on the hardware specs of the device this project is running on.
6. Finally, once downloading and training are successfully wrapped up with, launch the Flask web interface by running the command ```python main.py --mode app``` in the Terminal, and copy paste the link it returns into a web browser.

# Output Example
<img width="477" height="766" alt="3 Upload FIle" src="https://github.com/user-attachments/assets/cd152149-b8c2-490f-a7f8-72642efe1768"/>
