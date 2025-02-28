# Finite State Machine (FSM) Framework for Pattern Recognition

## Project Overview
This project, developed for ENSAM Casablanca, implements a Finite State Machine (FSM) to recognize patterns in complex data such as character sequences, images, and temporal sequences. Utilizing principles from FSM and machine learning techniques, this framework detects and classifies patterns in real datasets.

### Mini Project: Email Pattern Recognition System
The mini-project involves a backend email management system that processes incoming emails to determine if they are spam or not and classifies emails containing images into categories using machine learning models.

## System Architecture
1. **Email Input Handling**: Handles various types of email inputs, some of which include images.
2. **Spam Detection**: Uses a Recurrent Neural Network (RNN) to predict whether an email is spam.
3. **Image Categorization**: Employs a Convolutional Neural Network (CNN) to classify emails with images into thematic categories.

## Directory Structure
- `/data`: Contains all processed data.
- `/models`: Stores current models; new models are created here based on new data.
- `/rawdata`: Contains raw images and text files. Data is preprocessed here and saved to `/rawdatatreaten`.
- `/tmp`: Used for storing outputs from the models during processing.

## Project Workflow
1. **Data Preprocessing**: Data from `/rawdata` is processed and prepared for modeling.
2. **Model Training and Fine-tuning**: The system is capable of fine-tuning models based on the incoming data stream.
3. **Prediction and Classification**: The system predicts the nature of emails and classifies them accordingly.
4. **Output Storage**: Predictions and classifications are stored in `/tmp` for further analysis or real-time decision making.

## Customization and Scalability
- The CNN model's architecture is adjustable, allowing changes in layers and the addition of more classes to improve user experience and search efficiency within the mail system.
- For a real-world email service provider, this system can automatically handle incoming data and update models continuously without manual intervention.

## Professors
- Pr. LAZAIZ
- Pr. KAMOUSS

## Getting Started
To set up and run this project:
1. Clone the repository to your local machine.
2. Ensure you have Python and necessary libraries installed (see requirements.txt for details).
3. Navigate to the project directory and run `main.ipynb` to start the system.

## Contributions and Maintenance
Contributions to this project are welcome. Please fork the repository and submit pull requests for any enhancements.
For maintenance and support, contact [Your Contact Information].

## License
This project is developed as part of an educational assignment at ENSAM Casablanca. Use and distribution are permitted for educational purposes only.

## Project Repository
For more details and the latest updates, visit the project repository on GitHub:
[FSM-AI](https://github.com/crazytajdine/FSM-AI)
