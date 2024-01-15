# Streamlit Regression Analysis App

This Streamlit application enables users to perform logistic regression analysis on uploaded datasets. It includes functionalities for data loading, logistic regression, and post-regression analysis.

## Getting Started

### Prerequisites

- Anaconda
- Visual Studio Code

### Installation

1. **Clone the Repository:**
   If you have git installed, clone the repository to your local machine. Otherwise, you can download the project as a zip file and extract it.

    git clone https://github.com/salmaaboumzrag/logistic_regression_app.git


2. **Set Up an Anaconda Environment:**
    Open Anaconda Prompt and navigate to your project directory:

        cd path/to/the/project

    
    Create a new Anaconda environment:

        conda create --name streamlit_app python=3.8


    Activate the environment: 

        conda activate streamlit_app


3. **Install Dependencies:**
    Install the required Python packages:

        pip install streamlit pandas scikit-learn seaborn matplotlib


### Running the Application

1. **Open the Project in Visual Studio Code:**
Launch Visual Studio Code and open the project folder.

2. **Run Streamlit:**
Open the integrated terminal in Visual Studio Code, ensure that the `streamlit_app` environment is activated, and run the application:

streamlit run app.py


3. **Access the Application:**
The Streamlit application should now be running locally. Access it through your web browser at the address provided in the terminal (usually `http://localhost:8501`).

## Features

- **Data Upload:** Users can upload their datasets in CSV format.
- **Logistic Regression:** Perform logistic regression analysis with feature and target variable selection.
- **Post-Regression Analysis:** View classification reports and confusion matrices to evaluate the model's performance.

## Authors

- ABOUMZRAG Salma
- KERSI Lotfi




