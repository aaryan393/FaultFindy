# FaultFindy

FaultFindy is a Flask web application for identifying defects in tires using deep learning models.

## Steps to Run

1. **Pre-trained Model (Optional):**
   - For convenience, you can directly use the pre-trained model `effnet_finetuned.pth` provided in the repository. This allows you to run the application without the need to train the models, which can be time-consuming.
   
2. **Model Training (If you want to train models yourself):**
   - Run `FaultFindy.ipynb` to train the PyTorch models (DenseNet, ResNet, EfficientNet) and save them. This notebook contains the code for model training and saving.

3. **Model Loading:**
   - Use `model.py` to load the saved PyTorch models, including the provided pre-trained model `effnet_finetuned.pth`. This file contains the code for loading the models.

4. **Flask App:**
   - Run `app.py` to start the FaultFindy web application. This is the main Flask application file.

## Folder Structure

- **Static Folder:** Contains static files, such as images used in the application. In this case, it includes `defective-tires-cause-accidents-min.jpg`.
- **Templates Folder:** Contains HTML templates for the Flask application. Currently, it includes `index.html`.
- **FaultFindy Project Report.pdf:** This is a PDF report detailing the FaultFindy project.
- **FaultFindy.ipynb:** A Jupyter Notebook containing model training code for three PyTorch models: DenseNet, ResNet, and EfficientNet. Also includes code for saving the trained models.
- **app.py:** The main Flask application file.
- **model.py:** Contains code for loading the saved PyTorch models.
- **effnet_finetuned.pth:** A pre-trained EfficientNet model for direct use without training.
- **LICENSE:** This project is licensed under the [MIT License](LICENSE).

## Additional Information

For more detailed instructions on each step and how to use the Flask app, refer to the documentation or contact the project contributors.
 file.

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
