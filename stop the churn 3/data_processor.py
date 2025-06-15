import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='mean')
        
    def preprocess_data(self, df):
        """Preprocess the input dataframe."""
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Impute numeric columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
        
        # Encode categorical columns
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
        
        # Scale numeric features
        if len(numeric_cols) > 0:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        
        return df
    
    def preprocess_single_row(self, row):
        """Preprocess a single row of data for real-time prediction."""
        # Convert to DataFrame if it's not already
        if isinstance(row, pd.Series):
            row = pd.DataFrame([row])
        
        # Handle missing values
        numeric_cols = row.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = row.select_dtypes(include=['object']).columns
        
        # Impute numeric columns
        if len(numeric_cols) > 0:
            row[numeric_cols] = self.imputer.transform(row[numeric_cols])
        
        # Encode categorical columns
        for col in categorical_cols:
            if col in self.label_encoders:
                row[col] = self.label_encoders[col].transform(row[col].astype(str))
        
        # Scale numeric features
        if len(numeric_cols) > 0:
            row[numeric_cols] = self.scaler.transform(row[numeric_cols])
        
        return row 