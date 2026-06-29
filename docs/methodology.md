# Data Science Methodology for the Waste Segregation Assistant (AI Grade XII Syllabus)

Based on the AI Student Handbook (Class XII), Unit 2 "Data Science Methodology: An Analytic Approach to Capstone Project"

The Data Science Methodology put forward by John Rollins, a Data Scientist at IBM Analytics, consists of 10 iterative steps across 5 modules:

## Module 1: From Problem to Approach
**Step 1: Business Understanding**
   - Problem: School waste is not always sorted correctly
   - Goal: Classify waste into recyclable, compost, and landfill using computer vision
   - Activities: Stakeholder interviews, problem definition, solution identification
   - Questions: "What problem are you trying to solve?" "How can data help?"

**Step 2: Analytic Approach**
   - Use computer vision for image classification
   - Choose predictive approach (classification) vs. descriptive vs. prescriptive
   - Activities: Determine which analytics type (Descriptive, Diagnostic, Predictive, Prescriptive)
   - Questions: "Do I need to find how much (Regression) or which category (Classification)?"

## Module 2: From Requirements to Collection  
**Step 3: Data Requirements**
   - Need labeled images of waste items from school environment
   - Categories: recyclable, compost, landfill
   - Activities: Define data content, formats, sources, cleanup steps
   - Questions: "What data is needed?" "Where can we collect it?"

**Step 4: Data Collection**
   - Capture images using webcam or phone camera
   - Use both primary (webcam) and secondary (synthetic) sources
   - Activities: Systematic data gathering with quality checks
   - Questions: "How do we collect enough representative data?"

## Module 3: From Understanding to Preparation
**Step 5: Data Understanding**
   - Check class balance, image quality, lighting, background noise
   - Activities: Descriptive statistics, visualization, assess representativeness
   - Questions: "Does this data represent our problem?"

**Step 6: Data Preparation**
   - Resize images, normalize pixel values, augment training data
   - Split into train and validation sets
   - Activities: Data cleaning, transformation, feature engineering
   - Questions: "How do we prepare data for modeling?"

## Module 4: From Modeling to Evaluation
**Step 7: Modeling**
   - Train a CNN classifier with transfer learning
   - Activities: Choose model architecture, train, validate
   - Questions: "What model architecture works best?"

**Step 8: Evaluation**
   - Use accuracy, confusion matrix, precision, recall, F1 score
   - Activities: Cross-validation, error analysis, model refinement
   - Questions: "How well does the model perform?"

## Module 5: From Deployment to Feedback
**Step 9: Deployment**
   - Deploy model in Streamlit app for real-time classification
   - Activities: Web app development, user interface design
   - Questions: "How do we make this available to users?"

**Step 10: Feedback and Iteration**
   - Collect mistakes from users, add more real data, retrain and improve
   - Activities: Monitoring, user feedback collection, model updates
   - Questions: "How can we improve the model based on user experience?"

## Project Integration
This methodology directly maps to our Capstone Project:
- **Practical Activity**: Real-world waste segregation problem solving
- **Hands-on, Team Discussion, Web search, Case studies**: Collaborative development approach
- **10 Steps**: Complete the full AI project lifecycle as outlined in the AI curriculum
- **Iterative Refinement**: Continuous improvement based on user feedback and performance metrics

## Learning Outcomes
Students will be able to:
1. Integrate Data Science Methodology steps into the Capstone Project
2. Identify the best way to represent a solution to a problem
3. Understand the importance of validating machine learning models
4. Use key evaluation metrics for various machine learning tasks

## Alignment with AI Syllabus
This project fulfills the AI Grade XII requirements by:
- Applying Data Science Methodology throughout the project lifecycle
- Using computer vision techniques from the AI curriculum
- Demonstrating end-to-end problem solving from business understanding to deployment
- Building a practical, real-world solution that addresses environmental challenges

## Generative AI Component
Add optional generative AI assistant for educational content. See `scripts/generate_tips.py`.
