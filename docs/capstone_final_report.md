# Capstone Project Final Report

## Title
Waste Segregation Assistant Using Computer Vision

## Team Members
- Aaron
- Nishanth
- Sidardh
- Safa
- Sidhiksha

## Project Duration
Mid May 2026 to November 2026

## 1. Introduction
Waste segregation is an important part of environmental sustainability. This project was developed to help students identify whether an item belongs to recyclable, compost, or landfill waste.

## 2. Problem Definition
Many schools do not have an easy way to guide students while disposing of waste. This project solves that by using an AI model that classifies waste from images.

## 3. Business Understanding
The goal was to create a practical and useful AI solution for the school environment. The system needed to be simple, accurate, and suitable for a Grade 12 capstone project.

## 4. Data Science Methodology
The project followed the 10-step Data Science Methodology from the AI handbook:

1. Business Understanding
2. Analytic Approach
3. Data Requirements
4. Data Collection
5. Data Understanding
6. Data Preparation
7. Modeling
8. Evaluation
9. Deployment
10. Feedback and Iteration

## 5. Data Collection
Images were collected for three classes:
- Recyclable
- Compost
- Landfill

A webcam-based collection script was created to support real image capture.

## 6. Data Preparation
Images were resized and normalized before training. Augmentation was used to make the model more robust.

## 7. Modeling
A CNN-based model using transfer learning was built with TensorFlow and Keras.

## 8. Evaluation
The model was evaluated using:
- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrix

## 9. Deployment
A Streamlit web app was built so users can upload an image or use the webcam for prediction.

## 10. Dashboard and Storytelling
A dashboard was created to show waste segregation trends and project impact in a visual way.

## 11. Generative AI Component
An optional generative AI component was added to create educational tips and awareness poster text.

## 12. Results
The system works as a prototype for waste classification and project presentation. The synthetic dataset was used for testing, and the project is ready to be improved with real school waste images.

## 13. Conclusion
This capstone project combines computer vision, data science, and storytelling to solve a real-world school problem. It demonstrates how AI can support better environmental habits.

## 14. Future Scope
- Collect more real images
- Improve model accuracy
- Add more waste categories
- Deploy on mobile devices
- Track school waste reduction over time

## 15. Team Contribution
- Aaron: Project planning, documentation, and model coordination
- Nishanth: Data collection support and testing
- Sidardh: Model experimentation and app support
- Safa: Dashboard and presentation support
- Sidhiksha: Research, validation, and documentation support
