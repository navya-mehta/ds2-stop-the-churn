import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_processor import DataProcessor
from model_trainer import ModelTrainer
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import plotly.figure_factory as ff
from streamlit_lottie import st_lottie
import requests

# Set page config
st.set_page_config(
    page_title="Stop the Churn",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.streamlit.io/',
        'Report a bug': None,
        'About': '# Stop the Churn\nA customer churn prediction application.'
    }
)

# Load custom CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = ModelTrainer()
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()
if 'data' not in st.session_state:
    st.session_state.data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Lottie animation URL (dashboard theme)
lottie_dashboard = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_49rdyysj.json")

# Sidebar with enhanced styling (no navigation)
with st.sidebar:
    st_lottie(lottie_dashboard, height=120, key="dashboard_lottie")
    st.markdown("""
        <div style='text-align: center; padding: 1rem;'>
            <h1 style='color: var(--primary-color); margin-bottom: 1rem;'>Stop the Churn</h1>
            <p style='color: var(--text-color);'>Upload your customer data to predict churn</p>
        </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'], key="main_uploader")

# Main area navigation
page = st.selectbox(
    "Go to section:",
    ["Overview", "Risk Analysis", "Feature Importance", "Real-time Prediction"],
    key="main_nav"
)

if uploaded_file is not None:
    try:
        with st.spinner('Loading and processing data...'):
            df = pd.read_csv(uploaded_file)
            st.session_state.data = df
            
            # Automatically convert 'churn' column to numeric if needed
            if 'churn' in df.columns:
                if df['churn'].dtype == object:
                    unique_vals = set(df['churn'].str.lower().unique())
                    if unique_vals == {'yes', 'no'} or unique_vals == {'no', 'yes'}:
                        df['churn'] = df['churn'].str.lower().map({'yes': 1, 'no': 0})
            
            # Data Preview with enhanced styling
            with st.expander("📊 Preview Uploaded Data", expanded=False):
                st.dataframe(
                    df.head(10),
                    use_container_width=True,
                    height=300
                )
            
            processed_data = st.session_state.data_processor.preprocess_data(df)
            
            if 'churn' in df.columns:
                y = df['churn']
                X = processed_data.drop('churn', axis=1) if 'churn' in processed_data.columns else processed_data
                
                # Enhanced Stats Summary Cards
                total_customers = len(df)
                churned = int(df['churn'].sum())
                churn_rate = churned / total_customers if total_customers > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("""
                        <div class='metric-card animate-fade-in'>
                            <h3>Total Customers</h3>
                            <h2 style='color: var(--primary-color);'>{}</h2>
                        </div>
                    """.format(total_customers), unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                        <div class='metric-card animate-fade-in'>
                            <h3>Churned</h3>
                            <h2 style='color: #ef5350;'>{}</h2>
                        </div>
                    """.format(churned), unsafe_allow_html=True)
                
                with col3:
                    st.markdown("""
                        <div class='metric-card animate-fade-in'>
                            <h3>Churn Rate</h3>
                            <h2 style='color: var(--secondary-color);'>{:.1%}</h2>
                        </div>
                    """.format(churn_rate), unsafe_allow_html=True)
                
                if 'contract_type' in df.columns and 'churn' in df.columns:
                    contract_stats = df.groupby('contract_type')['churn'].mean().reset_index()
                    st.markdown("#### Churn Rate by Contract Type")
                    st.dataframe(contract_stats)
                    fig = px.bar(contract_stats, x='contract_type', y='churn', labels={'churn': 'Churn Rate'})
                    st.plotly_chart(fig, use_container_width=True)
                
                # Add Churn Rate by Segment section (after main metrics, before navigation switch)
                if uploaded_file is not None and 'churn' in df.columns:
                    st.markdown("### Churn Rate by Segment")
                    segment_columns = []
                    for col in ['gender', 'contract_type', 'payment_method']:
                        if col in df.columns:
                            segment_columns.append(col)
                    for col in segment_columns:
                        st.markdown(f"**Churn Rate by {col.replace('_', ' ').title()}**")
                        seg_stats = df.groupby(col)['churn'].mean().reset_index()
                        seg_stats['churn'] = seg_stats['churn'].apply(lambda x: f"{x:.1%}")
                        st.dataframe(seg_stats, use_container_width=True)
                        fig = px.bar(seg_stats, x=col, y='churn', labels={'churn': 'Churn Rate'}, title=f"Churn Rate by {col.replace('_', ' ').title()}")
                        st.plotly_chart(fig, use_container_width=True)
                
                with st.spinner('Training model and generating predictions...'):
                    results = st.session_state.model.train(X, y)
                    st.session_state.predictions = st.session_state.model.predict(X)
                    
                    # Model Evaluation with enhanced visualization
                    st.markdown("""
                        <div style='text-align: center; margin: 2rem 0;'>
                            <h2>Model Performance Metrics</h2>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    eval_col1, eval_col2 = st.columns(2)
                    with eval_col1:
                        st.markdown("""
                            <div class='stCard'>
                                <h3 style='text-align: center;'>Confusion Matrix</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        y_pred = (st.session_state.predictions > 0.5).astype(int)
                        cm = confusion_matrix(y, y_pred)
                        fig_cm = ff.create_annotated_heatmap(
                            cm,
                            x=["Predicted 0", "Predicted 1"],
                            y=["Actual 0", "Actual 1"],
                            colorscale='Blues',
                            showscale=True
                        )
                        fig_cm.update_layout(
                            template='plotly_white',
                            paper_bgcolor='white',
                            plot_bgcolor='white'
                        )
                        st.plotly_chart(fig_cm, use_container_width=True)
                    
                    with eval_col2:
                        st.markdown("""
                            <div class='stCard'>
                                <h3 style='text-align: center;'>ROC Curve</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        fpr, tpr, _ = roc_curve(y, st.session_state.predictions)
                        roc_auc = auc(fpr, tpr)
                        fig_roc = go.Figure()
                        fig_roc.add_trace(go.Scatter(
                            x=fpr, y=tpr,
                            name=f'ROC curve (AUC = {roc_auc:.2f})',
                            line=dict(color='#1E88E5', width=2)
                        ))
                        fig_roc.add_trace(go.Scatter(
                            x=[0, 1], y=[0, 1],
                            name='Random',
                            line=dict(color='#666', width=2, dash='dash')
                        ))
                        fig_roc.update_layout(
                            title='Receiver Operating Characteristic',
                            xaxis_title='False Positive Rate',
                            yaxis_title='True Positive Rate',
                            showlegend=True,
                            template='plotly_white',
                            paper_bgcolor='white',
                            plot_bgcolor='white'
                        )
                        st.plotly_chart(fig_roc, use_container_width=True)
                    
                    # Main content with tabs
                    st.markdown("""
                        <div style='text-align: center; margin: 2rem 0;'>
                            <h1>Churn Prediction Dashboard</h1>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Only show the selected section
                    if page == "Overview":
                        # Overview content (charts, metrics)
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("""
                                <div class='stCard'>
                                    <h3 style='text-align: center;'>Churn Probability Distribution</h3>
                                </div>
                            """, unsafe_allow_html=True)
                            fig = px.histogram(
                                x=st.session_state.predictions,
                                nbins=50,
                                title="Distribution of Churn Probabilities",
                                labels={'x': 'Churn Probability', 'y': 'Count'},
                                template='plotly_white'
                            )
                            fig.update_layout(
                                paper_bgcolor='white',
                                plot_bgcolor='white'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.markdown("""
                                <div class='stCard'>
                                    <h3 style='text-align: center;'>Churn vs Retain Distribution</h3>
                                </div>
                            """, unsafe_allow_html=True)
                            churn_count = (st.session_state.predictions > 0.5).sum()
                            retain_count = len(st.session_state.predictions) - churn_count
                            fig = px.pie(
                                values=[churn_count, retain_count],
                                names=['Churn', 'Retain'],
                                title="Churn vs Retain Distribution",
                                template='plotly_white',
                                color_discrete_sequence=['#ef5350', '#66bb6a']
                            )
                            fig.update_layout(
                                paper_bgcolor='white',
                                plot_bgcolor='white'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    elif page == "Risk Analysis":
                        # ... move tab2 content here ...
                        risk_categories = [st.session_state.model.get_risk_category(p) for p in st.session_state.predictions]
                        risk_df = pd.DataFrame({
                            'Customer ID': df.index,
                            'Churn Probability': st.session_state.predictions,
                            'Risk Category': risk_categories
                        })
                        
                        st.markdown("""
                            <div class='stCard'>
                                <h3 style='text-align: center;'>Customer Risk Distribution</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        risk_counts = pd.Series(risk_categories).value_counts()
                        fig = px.pie(
                            values=risk_counts.values,
                            names=risk_counts.index,
                            title="Customer Risk Distribution",
                            template='plotly_white',
                            color_discrete_sequence=['#ef5350', '#ffa726', '#66bb6a']
                        )
                        fig.update_layout(
                            paper_bgcolor='white',
                            plot_bgcolor='white'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("""
                            <div class='stCard'>
                                <h3 style='text-align: center;'>Top 10 High-Risk Customers</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        high_risk_df = risk_df[risk_df['Risk Category'] == 'High Risk'].sort_values('Churn Probability', ascending=False).head(10)
                        st.dataframe(
                            high_risk_df.style.background_gradient(subset=['Churn Probability'], cmap='RdYlGn_r'),
                            use_container_width=True
                        )
                    elif page == "Feature Importance":
                        # ... move tab3 content here ...
                        st.markdown("""
                            <div class='stCard'>
                                <h3 style='text-align: center;'>Feature Importance Analysis</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        importance_df = st.session_state.model.get_feature_importance()
                        fig = px.bar(
                            importance_df.head(10),
                            x='feature',
                            y='importance',
                            title="Top 10 Most Important Features",
                            template='plotly_white',
                            color='importance',
                            color_continuous_scale='Viridis'
                        )
                        fig.update_layout(
                            paper_bgcolor='white',
                            plot_bgcolor='white'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("""
                            <div class='stCard'>
                                <h3 style='text-align: center;'>SHAP Values Analysis</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        shap_values = st.session_state.model.get_shap_values(X)
                        fig, ax = plt.subplots()
                        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
                        st.pyplot(fig)
                    elif page == "Real-time Prediction":
                        # ... move tab4 content here ...
                        st.markdown("""
                            <div class='stCard'>
                                <h3 style='text-align: center;'>Real-time Prediction</h3>
                                <p style='text-align: center;'>Enter customer details for real-time prediction</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        input_data = {}
                        cols = st.columns(3)
                        for i, col in enumerate(df.columns):
                            if col != 'churn':
                                with cols[i % 3]:
                                    if df[col].dtype in ['int64', 'float64']:
                                        input_data[col] = st.number_input(
                                            f"Enter {col}",
                                            value=float(df[col].mean()),
                                            format="%.2f"
                                        )
                                    else:
                                        input_data[col] = st.selectbox(
                                            f"Select {col}",
                                            options=df[col].unique()
                                        )
                        
                        if st.button("Predict", key="predict_button"):
                            with st.spinner('Generating prediction...'):
                                input_df = pd.DataFrame([input_data])
                                processed_input = st.session_state.data_processor.preprocess_single_row(input_df)
                                prediction = st.session_state.model.predict(processed_input)[0]
                                risk_category = st.session_state.model.get_risk_category(prediction)
                                
                                st.markdown("""
                                    <div class='stCard'>
                                        <h3 style='text-align: center;'>Prediction Result</h3>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown(f"""
                                        <div class='metric-card'>
                                            <h3>Churn Probability</h3>
                                            <h2 style='color: var(--primary-color);'>{prediction:.2%}</h2>
                                        </div>
                                    """, unsafe_allow_html=True)
                                
                                with col2:
                                    st.markdown(f"""
                                        <div class='metric-card'>
                                            <h3>Risk Category</h3>
                                            <h2 class='risk-{risk_category.lower().split()[0]}'>{risk_category}</h2>
                                        </div>
                                    """, unsafe_allow_html=True)
                                
                                st.markdown("""
                                    <div class='stCard'>
                                        <h3 style='text-align: center;'>Feature Impact Analysis</h3>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                shap_values = st.session_state.model.get_shap_values(processed_input)
                                fig, ax = plt.subplots()
                                shap.force_plot(
                                    st.session_state.model.explainer.expected_value,
                                    shap_values[0],
                                    processed_input,
                                    matplotlib=True,
                                    show=False
                                )
                                st.pyplot(fig)
                    # Replace the upper download section with a static card header
                    st.markdown("""
                        <div class='stCard' style='margin-bottom: 1rem;'>
                            <h3 style='text-align: center; margin: 0;'>Download Predictions</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    predictions_df = pd.DataFrame({
                        'Customer ID': df.index,
                        'Churn Probability': st.session_state.predictions,
                        'Risk Category': [st.session_state.model.get_risk_category(p) for p in st.session_state.predictions]
                    })
                    
                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=predictions_df.to_csv(index=False),
                        file_name="predictions.csv",
                        mime="text/csv",
                        key="download_button"
                    )
            else:
                st.error("The uploaded CSV does not contain a 'churn' column. Please include the target column for training.")
    except Exception as e:
        st.error(f"Error processing file: {e}")
        import traceback
        st.text(traceback.format_exc())
        st.stop() 