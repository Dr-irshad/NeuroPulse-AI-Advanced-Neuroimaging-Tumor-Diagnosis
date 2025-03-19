# NeuroPulse AI: LLM-Enhanced Neuroimaging & Tumor Diagnosis

**Customer**: Wisio, Hong Kong  

## Overview

The **Brain Tumor Diagnosis System** is a specialized system developed to diagnose brain tumors through the use of annotated medical imaging data. The project incorporates advanced segmentation techniques to accurately classify tumor types such as glioma, meningioma, and pituitary tumors. The system is capable of detecting tumor regions at the pixel level, ensuring precise and reliable diagnostic support for medical professionals.

This system utilizes a dataset consisting of 675 images (original and augmented), allowing for robust training and testing of the model. The aim is to improve diagnostic accuracy and aid healthcare providers in detecting and classifying brain tumors in MRI scans.

## Key Features

- **Advanced Tumor Classification**: Classifies brain tumors into three categories:
  - Glioma
  - Meningioma
  - Pituitary Tumors
- **Pixel-Level Segmentation**: Provides accurate tumor region detection using YOLO segmentation.
- **Data Augmentation**: Enhances the model's training process through data augmentation, ensuring generalization to unseen data.
- **High Diagnostic Reliability**: Aims to improve diagnostic accuracy by leveraging deep learning and computer vision techniques.

## Dataset

The dataset used for training and testing includes 675 annotated MRI images (both original and augmented) that represent various types of brain tumors. The pixel-level annotations ensure precise tumor region identification. The dataset is divided into training and testing sets to validate the model's performance.

### Dataset Summary:

- **Total Images**: 675
  - **Glioma**: 250 images
  - **Meningioma**: 200 images
  - **Pituitary Tumors**: 125 images
- **Augmented Images**: 100 images
- **Annotations**: Pixel-level tumor region annotations

## Technologies Used

- **Computer Vision**: Used for analyzing MRI images and identifying tumor regions.
- **YOLO Segmentation**: Employed for accurate pixel-level tumor segmentation in the images.
- **Data Augmentation**: Techniques used to artificially increase the size of the training dataset by generating variations of the original images.
- **Streamlit UI:** A user-friendly interface for uploading images/videos and running the tumor detection model.
- **Inference API:** Utilizes Roboflow's detection API for real-time tumor analysis.
- **Ollama-powered LLM:** DeepSeek-R1 is used to generate medical insights and explanations based on detection results.
  
## Installation

Follow the steps below to set up the Brain Tumor Diagnosis System on your local machine.

### Prerequisites

Before installing the project, make sure you have the following dependencies installed:

- Python 3.x
- pip (Python package installer)
- Streamlit
- OpenCV
- NumPy
- PIL
- InferenceHTTPClient (Roboflow API)
- LangChain & Ollama (for AI-powered tumor analysis)

### Clone the repository

    ```bash
    git clone https://github.com/Dr-irshad/NeuroPulse AI: LLM-Enhanced Neuroimaging & Tumor Diagnosis.git
    cd brain-tumor-diagnosis-system

### Install dependencies
Install the required Python packages by running:
    ```bash
    pip install -r requirements.txt

### Additional Setup
  1. Download the dataset and place it in the data/ folder (follow the provided instructions or link to access the dataset).
  2. Ensure that the annotations and augmented data are correctly placed in the appropriate folders.
  3. Install and configure Ollama for LLM-powered tumor analysis.

## Usage

### Running the Streamlit UI
To launch the user interface, run the following command:
  
    ```bash
    streamlit run app.py 

This will open a web-based UI where users can upload MRI images or videos for analysis.

### Training the Model
To start training the YOLO segmentation model on the dataset, run the following command:

    ```bash
    python train.py --data_path ./data --epochs 50 --batch_size 16

This command will train the model for 50 epochs with a batch size of 16. You can adjust the hyperparameters as needed.

### Testing the Model
Once training is complete, you can test the model on new images by running:

      ```bash
      python test.py --model_path ./model_weights/best_model.pth --test_images ./test_images/

This will load the trained model and run predictions on the test images stored in the test_images/ folder.

### Model Evaluation
The system provides evaluation metrics such as accuracy, precision, recall, and F1-score to assess the model's performance on the test dataset.
    
    ```bash
    python evaluate.py --model_path ./model_weights/best_model.pth --test_images ./test_images/

### Results
The model is evaluated based on the following metrics:

- **Accuracy:** Percentage of correct tumor classifications.
- **Precision:** Precision of the tumor region identification.
- **Recall:** Ability of the model to correctly detect tumors.
- **F1-Score:** Harmonic mean of precision and recall.
  
The system has shown improved diagnostic reliability and tumor classification accuracy in comparison to traditional methods, especially when combined with data augmentation techniques.

## User Interface

Below is a preview of the system's UI:

![Brain Tumor Detection UI](output/Brain-tumor.gif)


## Contributing
If you would like to contribute to this project, please follow these steps:

  - Fork the repository.
  - Create a new branch (git checkout -b feature-name).
  - Commit your changes (git commit -am 'Add feature').
  - Push to the branch (git push origin feature-name).
  - Open a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
The dataset used for this project is publicly available from various medical imaging sources.
Special thanks to the team at Wisio, Hong Kong, for the collaboration and support in developing this system.
