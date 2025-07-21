#!/usr/bin/env python3
"""
Main launcher for Stock Tracking Streamlit Application
Run this file to automatically start the Streamlit app
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path


def find_streamlit_executable():
    """Find streamlit executable in various locations"""
    # Try common locations for streamlit
    possible_locations = [
        "streamlit",  # If in PATH
        sys.executable.replace("python.exe", "streamlit.exe"),  # Same folder as python
        sys.executable.replace("python.exe", "Scripts/streamlit.exe"),  # Scripts folder
        os.path.join(os.path.dirname(sys.executable), "streamlit.exe"),
        os.path.join(os.path.dirname(sys.executable), "Scripts", "streamlit.exe"),
    ]

    for location in possible_locations:
        try:
            # Test if streamlit executable exists and works
            result = subprocess.run([location, "--version"],
                                    capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"✅ Found Streamlit at: {location}")
                return location
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            continue

    return None


def install_streamlit():
    """Install streamlit if not found"""
    print("📦 Streamlit not found. Installing...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
        print("✅ Streamlit installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Streamlit: {e}")
        return False


def check_dependencies():
    """Check and install required dependencies"""
    required_packages = [
        "streamlit",
        "pandas",
        "numpy",
        "plotly",
        "scikit-learn"
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"📦 Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([
                                      sys.executable, "-m", "pip", "install"
                                  ] + missing_packages)
            print("✅ All dependencies installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False

    # Try to install Prophet (optional)
    try:
        import prophet
        print("✅ Prophet is available")
    except ImportError:
        print("📦 Installing Prophet (optional for advanced predictions)...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "prophet"])
            print("✅ Prophet installed successfully!")
        except subprocess.CalledProcessError:
            print("⚠️ Prophet installation failed. Using linear regression fallback.")

    return True


def create_streamlit_app():
    """Create the streamlit app file if it doesn't exist"""
    app_filename = "streamlit_app.py"

    if not os.path.exists(app_filename):
        print(f"📝 Creating {app_filename}...")

        # The complete Streamlit app code
        app_code = '''import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import Prophet, fallback to linear regression if not available
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    from sklearn.linear_model import LinearRegression
    PROPHET_AVAILABLE = False
    st.warning("Prophet not available. Using linear regression for predictions.")

def load_and_validate_data(uploaded_file):
    """
    Load and validate the uploaded CSV file
    Expected columns: date, product, entry, exit, current_stock
    """
    try:
        df = pd.read_csv(uploaded_file)

        # Check required columns
        required_cols = ['date', 'product', 'entry', 'exit', 'current_stock']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            st.info("Required columns: date, product, entry, exit, current_stock")
            return None

        # Convert date column
        df['date'] = pd.to_datetime(df['date'])

        # Ensure numeric columns
        numeric_cols = ['entry', 'exit', 'current_stock']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Remove rows with missing data
        df = df.dropna()

        return df

    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def predict_stock_prophet(df_product, days=30):
    """
    Predict stock levels using Prophet
    """
    # Prepare data for Prophet
    prophet_df = df_product[['date', 'current_stock']].rename(columns={'date': 'ds', 'current_stock': 'y'})

    # Initialize and fit Prophet model
    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=False)
    model.fit(prophet_df)

    # Create future dataframe
    future = model.make_future_dataframe(periods=days)

    # Make predictions
    forecast = model.predict(future)

    # Get predictions for the next 30 days
    last_date = df_product['date'].max()
    future_predictions = forecast[forecast['ds'] > last_date][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    return future_predictions.rename(columns={'ds': 'date', 'yhat': 'predicted_stock'})

def predict_stock_linear(df_product, days=30):
    """
    Predict stock levels using linear regression (fallback method)
    """
    # Prepare data
    df_product = df_product.sort_values('date')
    df_product['days_since_start'] = (df_product['date'] - df_product['date'].min()).dt.days

    # Fit linear regression
    X = df_product[['days_since_start']]
    y = df_product['current_stock']

    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X, y)

    # Create future dates
    last_date = df_product['date'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
    future_days = [(date - df_product['date'].min()).days for date in future_dates]

    # Make predictions
    predictions = model.predict([[day] for day in future_days])

    # Create prediction dataframe
    future_predictions = pd.DataFrame({
        'date': future_dates,
        'predicted_stock': predictions
    })

    return future_predictions

def create_stock_chart(df_product, predictions, product_name):
    """
    Create interactive stock chart with predictions
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'Stock Level - {product_name}', 'Daily Entry/Exit'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )

    # Stock level chart
    fig.add_trace(
        go.Scatter(
            x=df_product['date'],
            y=df_product['current_stock'],
            mode='lines+markers',
            name='Actual Stock',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ),
        row=1, col=1
    )

    # Predictions
    if predictions is not None and not predictions.empty:
        fig.add_trace(
            go.Scatter(
                x=predictions['date'],
                y=predictions['predicted_stock'],
                mode='lines+markers',
                name='Predicted Stock',
                line=dict(color='red', width=2, dash='dash'),
                marker=dict(size=4)
            ),
            row=1, col=1
        )

        # Add confidence interval if available (Prophet)
        if 'yhat_lower' in predictions.columns:
            fig.add_trace(
                go.Scatter(
                    x=predictions['date'].tolist() + predictions['date'].tolist()[::-1],
                    y=predictions['yhat_upper'].tolist() + predictions['yhat_lower'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.1)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=False,
                    name='Confidence Interval'
                ),
                row=1, col=1
            )

    # Entry/Exit chart
    fig.add_trace(
        go.Bar(
            x=df_product['date'],
            y=df_product['entry'],
            name='Entry',
            marker_color='green',
            opacity=0.7
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Bar(
            x=df_product['date'],
            y=-df_product['exit'],
            name='Exit',
            marker_color='red',
            opacity=0.7
        ),
        row=2, col=1
    )

    # Add critical threshold line
    critical_threshold = st.session_state.get('critical_threshold', 20)
    fig.add_hline(
        y=critical_threshold,
        line_dash="dot",
        line_color="orange",
        annotation_text=f"Critical Level ({critical_threshold})",
        row=1, col=1
    )

    fig.update_layout(
        height=600,
        showlegend=True,
        title=f"Stock Analysis - {product_name}",
        hovermode='x unified'
    )

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Stock Quantity", row=1, col=1)
    fig.update_yaxes(title_text="Entry/Exit", row=2, col=1)

    return fig

def check_critical_alerts(predictions, critical_threshold=20):
    """
    Check for critical stock level alerts
    """
    alerts = []

    if predictions is not None and not predictions.empty:
        critical_days = predictions[predictions['predicted_stock'] < critical_threshold]

        if not critical_days.empty:
            first_critical_day = critical_days.iloc[0]
            min_stock_day = predictions.loc[predictions['predicted_stock'].idxmin()]

            alerts.append({
                'type': 'critical',
                'message': f"⚠️ Stock will fall below critical level ({critical_threshold}) on {first_critical_day['date'].strftime('%Y-%m-%d')}",
                'days_until': (first_critical_day['date'] - datetime.now()).days
            })

            alerts.append({
                'type': 'minimum',
                'message': f"📉 Minimum predicted stock: {min_stock_day['predicted_stock']:.1f} units on {min_stock_day['date'].strftime('%Y-%m-%d')}",
                'value': min_stock_day['predicted_stock']
            })

    return alerts

def main():
    st.set_page_config(
        page_title="Stock Tracking & Prediction",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 Stock Tracking & Prediction System")
    st.markdown("Upload your stock data CSV file and get insights with 30-day predictions")

    # Sidebar configuration
    st.sidebar.header("Configuration")

    # Critical threshold setting
    critical_threshold = st.sidebar.slider(
        "Critical Stock Threshold",
        min_value=1,
        max_value=100,
        value=20,
        help="Alert when stock is predicted to fall below this level"
    )
    st.session_state['critical_threshold'] = critical_threshold

    # Prediction days
    prediction_days = st.sidebar.slider(
        "Prediction Period (days)",
        min_value=7,
        max_value=90,
        value=30,
        help="Number of days to predict into the future"
    )

    # File upload
    st.header("📁 Data Upload")

    uploaded_file = st.file_uploader(
        "Upload your stock data CSV file",
        type=['csv'],
        help="Required columns: date, product, entry, exit, current_stock"
    )

    # Sample data format
    with st.expander("📋 Expected CSV Format"):
        sample_data = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'product': ['Product A', 'Product A', 'Product B'],
            'entry': [100, 50, 75],
            'exit': [30, 40, 25],
            'current_stock': [170, 180, 125]
        })
        st.dataframe(sample_data)

    if uploaded_file is not None:
        # Load and validate data
        df = load_and_validate_data(uploaded_file)

        if df is not None:
            st.success(f"✅ Data loaded successfully! {len(df)} records found")

            # Data overview
            st.header("📈 Data Overview")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Products", df['product'].nunique())
            with col2:
                st.metric("Date Range", f"{(df['date'].max() - df['date'].min()).days} days")
            with col3:
                st.metric("Total Entries", f"{df['entry'].sum():,.0f}")
            with col4:
                st.metric("Total Exits", f"{df['exit'].sum():,.0f}")

            # Product selection
            products = sorted(df['product'].unique())
            selected_product = st.selectbox("Select Product for Analysis", products)

            # Filter data for selected product
            df_product = df[df['product'] == selected_product].sort_values('date')

            if len(df_product) < 2:
                st.error("Selected product needs at least 2 data points for analysis")
                return

            # Generate predictions
            st.header("🔮 Stock Predictions")

            with st.spinner("Generating predictions..."):
                try:
                    if PROPHET_AVAILABLE and len(df_product) >= 10:
                        predictions = predict_stock_prophet(df_product, prediction_days)
                        st.info("Using Prophet model for predictions")
                    else:
                        predictions = predict_stock_linear(df_product, prediction_days)
                        st.info("Using linear regression for predictions")

                except Exception as e:
                    st.error(f"Error generating predictions: {str(e)}")
                    predictions = None

            # Display chart
            if predictions is not None:
                fig = create_stock_chart(df_product, predictions, selected_product)
                st.plotly_chart(fig, use_container_width=True)

                # Alerts
                alerts = check_critical_alerts(predictions, critical_threshold)

                if alerts:
                    st.header("🚨 Alerts")
                    for alert in alerts:
                        if alert['type'] == 'critical':
                            st.error(alert['message'])
                        else:
                            st.warning(alert['message'])

                # Prediction summary
                st.header("📊 Prediction Summary")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Current Status")
                    current_stock = df_product['current_stock'].iloc[-1]
                    last_date = df_product['date'].max().strftime('%Y-%m-%d')

                    st.metric(
                        f"Current Stock ({last_date})",
                        f"{current_stock:.0f} units"
                    )

                with col2:
                    st.subheader("30-Day Outlook")
                    final_prediction = predictions['predicted_stock'].iloc[-1]
                    trend = "📈" if final_prediction > current_stock else "📉"

                    st.metric(
                        f"Predicted Stock (30 days)",
                        f"{final_prediction:.0f} units",
                        f"{final_prediction - current_stock:.0f} units {trend}"
                    )

                # Detailed predictions table
                with st.expander("📋 Detailed Predictions"):
                    display_predictions = predictions.copy()
                    display_predictions['predicted_stock'] = display_predictions['predicted_stock'].round(1)
                    display_predictions['date'] = display_predictions['date'].dt.strftime('%Y-%m-%d')
                    st.dataframe(display_predictions, use_container_width=True)

            # Raw data view
            with st.expander("📊 Raw Data"):
                st.dataframe(df_product, use_container_width=True)

    else:
        st.info("👆 Please upload a CSV file to begin analysis")

if __name__ == "__main__":
    main()
'''

        with open(app_filename, 'w', encoding='utf-8') as f:
            f.write(app_code)

        print(f"✅ Created {app_filename}")

    return app_filename


def main():
    """Main function to launch the Streamlit application"""
    print("🚀 Stock Tracking Application Launcher")
    print("=" * 50)

    # Check and install dependencies
    print("📋 Checking dependencies...")
    if not check_dependencies():
        print("❌ Failed to install dependencies. Please install manually.")
        return

    # Create streamlit app if it doesn't exist
    app_file = create_streamlit_app()

    # Find streamlit executable
    streamlit_path = find_streamlit_executable()

    if not streamlit_path:
        print("❌ Streamlit not found. Attempting to install...")
        if install_streamlit():
            streamlit_path = find_streamlit_executable()

        if not streamlit_path:
            print("❌ Could not find or install Streamlit.")
            print("💡 Try installing manually with: pip install streamlit")
            return

    # Launch Streamlit app
    print(f"🌐 Launching Streamlit app: {app_file}")
    print("📍 The app will open in your default browser...")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)

    try:
        # Run streamlit
        subprocess.run([streamlit_path, "run", app_file, "--server.headless", "false"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running Streamlit: {e}")
        print("💡 Try running manually: streamlit run streamlit_app.py")


if __name__ == "__main__":
    main()