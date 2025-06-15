  # Stop the Churn - Customer Churn Prediction System

A comprehensive churn prediction system with an interactive Streamlit dashboard that helps identify customers at risk of churning.

## Features

- Upload and process customer data in CSV format
- Automated data preprocessing and feature engineering
- Binary classification model for churn prediction
- Interactive dashboard with key visualizations
- Real-time prediction capabilities
- Feature importance analysis using SHAP values
- Dark mode support
- Mobile-friendly interface

## Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/stop-the-churn.git
cd stop-the-churn
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

## Usage

1. Launch the application using the command above
2. Upload your customer data CSV file
3. The system will automatically:
   - Process and clean the data
   - Train the prediction model
   - Generate visualizations
   - Provide churn predictions

## Dashboard Features

- Distribution plot of churn probabilities
- Churn vs Retain pie chart
- Top-10 high-risk customers table
- Downloadable predictions file
- Feature importance visualization
- Real-time single-customer prediction

## Model Performance

The system uses a binary classifier optimized for AUC-ROC score to predict customer churn within 30 days.

## Contributing

Feel free to submit issues and enhancement requests! 