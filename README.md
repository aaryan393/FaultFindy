# FaultFindy

FaultFindy is a Flask web application for identifying defects in tires using deep learning models.

## Steps to Run

1. **Model Training:** 
   - First, run `FaultFindy.ipynb` to train the PyTorch models (DenseNet, ResNet, EfficientNet) and save them. This notebook contains the code for model training and saving.

2. **Model Loading:**
   - After saving the models, use `model.py` to load the saved PyTorch models. This file contains the code for loading the models.

3. **Flask App:**
   - Finally, run `app.py` to start the FaultFindy web application. This is the main Flask application file.

## Static Folder

Contains static files, such as images used in the application. In this case, it includes `defective-tires-cause-accidents-min.jpg`.

## Templates Folder

Contains HTML templates for the Flask application. Currently, it includes `index.html`.

## FaultFindy Project Report

This is a PDF report detailing the FaultFindy project.

## FaultFindy.ipynb

A Jupyter Notebook containing model training code for three PyTorch models: DenseNet, ResNet, and EfficientNet. Also includes code for saving the trained models.

## app.py

This is the main Flask application file. Run this file to start the FaultFindy web application.

## model.py

Contains code for loading the saved PyTorch models.

## License

This project is licensed under the [MIT License](LICENSE).

For more information on how to run the Flask app and utilize the models, refer to the documentation or contact the project contributors.
