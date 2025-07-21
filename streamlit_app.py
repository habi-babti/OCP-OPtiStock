import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import requests
import json
import logging
from typing import Optional, Dict, List, Tuple
import time
from dataclasses import dataclass
from pathlib import Path
import io

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import Prophet, fallback to linear regression if not available
try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except ImportError:
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    PROPHET_AVAILABLE = False


@dataclass
class StockAlert:
    """Data class for stock alerts"""
    type: str
    message: str
    severity: str  # 'critical', 'warning', 'info'
    days_until: Optional[int] = None
    value: Optional[float] = None


@dataclass
class PredictionResult:
    """Data class for prediction results"""
    predictions: pd.DataFrame
    model_type: str
    accuracy_score: Optional[float] = None
    confidence_interval: Optional[pd.DataFrame] = None


class ConfigManager:
    """Manage application configuration and settings"""

    @staticmethod
    def get_default_config() -> Dict:
        return {
            'critical_threshold': 20,
            'warning_threshold': 50,
            'prediction_days': 30,
            'cache_duration': 3600,  # 1 hour
            'max_file_size': 50,  # MB
            'supported_formats': ['csv', 'xlsx', 'json'],
            'ai_timeout': 60,
            'chart_theme': 'plotly_white'
        }

    @staticmethod
    def load_config() -> Dict:
        """Load configuration from session state or defaults"""
        config = ConfigManager.get_default_config()

        # Update with session state values if they exist
        for key in config:
            if f"config_{key}" in st.session_state:
                config[key] = st.session_state[f"config_{key}"]

        return config


class DataValidator:
    """Enhanced data validation and preprocessing"""

    @staticmethod
    def validate_data_structure(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate the basic structure of the dataframe"""
        required_cols = ['date', 'product', 'entry', 'exit', 'current_stock']
        errors = []

        # Check required columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {', '.join(missing_cols)}")

        # Check data types and ranges
        if 'date' in df.columns:
            try:
                pd.to_datetime(df['date'])
            except Exception as e:
                errors.append(f"Invalid date format: {str(e)}")

        # Check numeric columns
        numeric_cols = ['entry', 'exit', 'current_stock']
        for col in numeric_cols:
            if col in df.columns:
                try:
                    numeric_data = pd.to_numeric(df[col], errors='coerce')
                    if numeric_data.isna().all():
                        errors.append(f"Column '{col}' contains no valid numeric data")
                    elif numeric_data.min() < 0:
                        errors.append(f"Column '{col}' contains negative values")
                except Exception as e:
                    errors.append(f"Error validating column '{col}': {str(e)}")

        return len(errors) == 0, errors

    @staticmethod
    def clean_and_preprocess(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the data"""
        df_cleaned = df.copy()

        # Convert date column
        df_cleaned['date'] = pd.to_datetime(df_cleaned['date'])

        # Ensure numeric columns
        numeric_cols = ['entry', 'exit', 'current_stock']
        for col in numeric_cols:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

        # Remove rows with missing critical data
        df_cleaned = df_cleaned.dropna(subset=['date', 'product', 'current_stock'])

        # Fill missing entry/exit with 0
        df_cleaned[['entry', 'exit']] = df_cleaned[['entry', 'exit']].fillna(0)

        # Sort by product and date
        df_cleaned = df_cleaned.sort_values(['product', 'date'])

        # Calculate derived metrics
        df_cleaned['net_change'] = df_cleaned['entry'] - df_cleaned['exit']
        df_cleaned['stock_velocity'] = df_cleaned.groupby('product')['current_stock'].pct_change()
        df_cleaned['days_since_start'] = df_cleaned.groupby('product')['date'].transform(
            lambda x: (x - x.min()).dt.days
        )

        return df_cleaned

    @staticmethod
    def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies in stock data"""
        df_anomalies = df.copy()

        for product in df['product'].unique():
            product_data = df[df['product'] == product]

            # Calculate z-scores for stock levels
            stock_mean = product_data['current_stock'].mean()
            stock_std = product_data['current_stock'].std()

            if stock_std > 0:
                z_scores = np.abs((product_data['current_stock'] - stock_mean) / stock_std)
                df_anomalies.loc[product_data.index, 'anomaly_score'] = z_scores
                df_anomalies.loc[product_data.index, 'is_anomaly'] = z_scores > 3
            else:
                df_anomalies.loc[product_data.index, 'anomaly_score'] = 0
                df_anomalies.loc[product_data.index, 'is_anomaly'] = False

        return df_anomalies


class LlamaAnalyzer:
    """Enhanced interface for Ollama Llama 3.1 local AI analysis"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.1"):
        self.base_url = base_url
        self.model = model
        self.session_cache = {}

    def is_available(self) -> bool:
        """Check if Ollama is running and Llama model is available"""
        if 'llama_status' in st.session_state:
            # Use cached status for 5 minutes
            last_check, status = st.session_state['llama_status']
            if time.time() - last_check < 300:  # 5 minutes
                return status

        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available = any(self.model in model.get('name', '') for model in models)
                st.session_state['llama_status'] = (time.time(), available)
                return available
            st.session_state['llama_status'] = (time.time(), False)
            return False
        except Exception as e:
            logger.warning(f"Llama availability check failed: {e}")
            st.session_state['llama_status'] = (time.time(), False)
            return False

    def generate_analysis(self, prompt: str, context_data: Optional[str] = None,
                          temperature: float = 0.3, max_tokens: int = 1000) -> str:
        """Generate analysis using Llama 3.1 with enhanced error handling"""
        try:
            # Check cache first
            cache_key = hash(prompt + (context_data or ""))
            if cache_key in self.session_cache:
                return self.session_cache[cache_key]

            # Prepare the full prompt with context
            if context_data:
                full_prompt = f"""
STOCK DATA ANALYSIS CONTEXT:
{context_data}

ANALYSIS REQUEST:
{prompt}

Please provide a structured analysis with:
1. **Key Observations**: Main trends and patterns
2. **Risk Assessment**: Potential issues or concerns
3. **Opportunities**: Positive trends or optimization areas
4. **Actionable Recommendations**: Specific steps to take
5. **Timeline**: When to implement recommendations

Keep the response focused, actionable, and data-driven.
"""
            else:
                full_prompt = prompt

            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": 0.9,
                    "max_tokens": max_tokens,
                    "stop": ["Human:", "Assistant:"]
                }
            }

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json().get('response', 'No response generated')
                # Cache successful responses
                self.session_cache[cache_key] = result
                return result
            else:
                return f"Error: {response.status_code} - {response.text}"

        except requests.exceptions.Timeout:
            return "⏱️ Analysis timeout - Llama model may be busy. Please try again."
        except Exception as e:
            logger.error(f"Llama analysis error: {e}")
            return f"🔌 Connection error: {str(e)}"

    def analyze_comprehensive(self, df_product: pd.DataFrame, predictions: Optional[pd.DataFrame] = None,
                              alerts: List[StockAlert] = None) -> str:
        """Generate comprehensive stock analysis"""
        # Calculate advanced metrics
        current_stock = df_product['current_stock'].iloc[-1]
        avg_entry = df_product['entry'].mean()
        avg_exit = df_product['exit'].mean()
        stock_trend = df_product['current_stock'].iloc[-1] - df_product['current_stock'].iloc[0]
        volatility = df_product['current_stock'].std()
        turnover_rate = df_product['exit'].sum() / df_product['current_stock'].mean() if df_product[
                                                                                             'current_stock'].mean() > 0 else 0

        # Recent performance (last 7 days)
        recent_data = df_product.tail(7)
        recent_trend = recent_data['current_stock'].iloc[-1] - recent_data['current_stock'].iloc[0] if len(
            recent_data) > 1 else 0

        context = f"""
PRODUCT: {df_product['product'].iloc[0]}
ANALYSIS PERIOD: {df_product['date'].min().strftime('%Y-%m-%d')} to {df_product['date'].max().strftime('%Y-%m-%d')}

CURRENT STATUS:
- Current Stock Level: {current_stock:.1f} units
- Stock Volatility: {volatility:.2f}
- Inventory Turnover Rate: {turnover_rate:.2f}

FLOW METRICS:
- Average Daily Entry: {avg_entry:.1f} units
- Average Daily Exit: {avg_exit:.1f} units
- Net Daily Change: {avg_entry - avg_exit:.1f} units

TREND ANALYSIS:
- Overall Trend: {'Increasing' if stock_trend > 0 else 'Decreasing'} by {abs(stock_trend):.1f} units
- Recent 7-day Trend: {'Increasing' if recent_trend > 0 else 'Decreasing'} by {abs(recent_trend):.1f} units

DATA QUALITY:
- Total Data Points: {len(df_product)}
- Date Range: {(df_product['date'].max() - df_product['date'].min()).days} days
"""

        if predictions is not None and len(predictions) > 0:
            pred_end = predictions['predicted_stock'].iloc[-1]
            pred_change = pred_end - current_stock
            context += f"""

PREDICTIONS (30-day forecast):
- Predicted Stock Level: {pred_end:.1f} units
- Expected Change: {pred_change:+.1f} units ({pred_change / current_stock * 100:+.1f}%)
"""

        if alerts:
            context += f"""

ACTIVE ALERTS:
{chr(10).join([f"- {alert.message}" for alert in alerts])}
"""

        prompt = """Analyze this inventory data and provide strategic insights for inventory management. 
        Focus on optimization opportunities, risk mitigation, and operational efficiency."""

        return self.generate_analysis(prompt, context)


class PredictionEngine:
    """Enhanced prediction engine with multiple models and validation"""

    @staticmethod
    def predict_with_prophet(df_product: pd.DataFrame, days: int = 30) -> PredictionResult:
        """Predict stock levels using Prophet with enhanced features"""
        try:
            # Prepare data for Prophet
            prophet_df = df_product[['date', 'current_stock']].rename(
                columns={'date': 'ds', 'current_stock': 'y'}
            )

            # Add regressors for entry and exit
            prophet_df['entry'] = df_product['entry'].values
            prophet_df['exit'] = df_product['exit'].values

            # Initialize Prophet with seasonality
            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=len(df_product) > 365,
                changepoint_prior_scale=0.05,
                interval_width=0.8
            )

            # Add regressors
            model.add_regressor('entry')
            model.add_regressor('exit')

            # Fit model
            model.fit(prophet_df)

            # Create future dataframe
            future = model.make_future_dataframe(periods=days)

            # Estimate future entry/exit based on recent averages
            recent_entry = df_product['entry'].tail(7).mean()
            recent_exit = df_product['exit'].tail(7).mean()

            future['entry'] = recent_entry
            future['exit'] = recent_exit

            # Make predictions
            forecast = model.predict(future)

            # Extract future predictions
            last_date = df_product['date'].max()
            future_predictions = forecast[forecast['ds'] > last_date][
                ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
            ].rename(columns={'ds': 'date', 'yhat': 'predicted_stock'})

            # Calculate confidence intervals
            confidence_df = forecast[forecast['ds'] > last_date][
                ['ds', 'yhat_lower', 'yhat_upper']
            ].rename(columns={'ds': 'date'})

            return PredictionResult(
                predictions=future_predictions,
                model_type="Prophet (Time Series)",
                confidence_interval=confidence_df
            )

        except Exception as e:
            logger.error(f"Prophet prediction error: {e}")
            return PredictionEngine.predict_with_regression(df_product, days)

    @staticmethod
    def predict_with_regression(df_product: pd.DataFrame, days: int = 30,
                                model_type: str = "ridge") -> PredictionResult:
        """Enhanced regression prediction with multiple algorithms"""
        try:
            # Prepare features
            df_product = df_product.sort_values('date')
            df_product['days_since_start'] = (df_product['date'] - df_product['date'].min()).dt.days
            df_product['moving_avg_3'] = df_product['current_stock'].rolling(window=3, min_periods=1).mean()
            df_product['moving_avg_7'] = df_product['current_stock'].rolling(window=7, min_periods=1).mean()

            # Features: days, moving averages, entry/exit patterns
            features = ['days_since_start', 'entry', 'exit', 'moving_avg_3']
            X = df_product[features].fillna(method='ffill')
            y = df_product['current_stock']

            # Choose model
            if model_type == "ridge":
                model = Ridge(alpha=1.0)
                model_name = "Ridge Regression"
            else:
                model = LinearRegression()
                model_name = "Linear Regression"

            # Fit model
            model.fit(X, y)

            # Calculate accuracy on training data
            y_pred_train = model.predict(X)
            accuracy = 1 - mean_absolute_error(y, y_pred_train) / y.mean()

            # Create future dates and features
            last_date = df_product['date'].max()
            last_features = X.iloc[-1].copy()

            future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
            future_features = []

            for i in range(days):
                future_row = last_features.copy()
                future_row['days_since_start'] += (i + 1)
                # Use recent averages for entry/exit
                future_row['entry'] = df_product['entry'].tail(7).mean()
                future_row['exit'] = df_product['exit'].tail(7).mean()
                future_features.append(future_row)

            future_X = pd.DataFrame(future_features, columns=features)

            # Make predictions
            predictions = model.predict(future_X)

            # Create prediction dataframe
            future_predictions = pd.DataFrame({
                'date': future_dates,
                'predicted_stock': predictions
            })

            return PredictionResult(
                predictions=future_predictions,
                model_type=model_name,
                accuracy_score=max(0, min(1, accuracy))
            )

        except Exception as e:
            logger.error(f"Regression prediction error: {e}")
            raise

    @staticmethod
    def predict_stock(df_product: pd.DataFrame, days: int = 30) -> PredictionResult:
        """Main prediction method with fallback logic"""
        if len(df_product) < 3:
            raise ValueError("Need at least 3 data points for prediction")

        try:
            if PROPHET_AVAILABLE and len(df_product) >= 10:
                return PredictionEngine.predict_with_prophet(df_product, days)
            else:
                return PredictionEngine.predict_with_regression(df_product, days, "ridge")
        except Exception as e:
            logger.warning(f"Primary prediction failed, using fallback: {e}")
            return PredictionEngine.predict_with_regression(df_product, days, "linear")


class AlertSystem:
    """Enhanced alert system for stock monitoring"""

    @staticmethod
    def generate_alerts(df_product: pd.DataFrame, predictions: Optional[pd.DataFrame] = None,
                        config: Dict = None) -> List[StockAlert]:
        """Generate comprehensive stock alerts"""
        if config is None:
            config = ConfigManager.get_default_config()

        alerts = []
        current_stock = df_product['current_stock'].iloc[-1]
        critical_threshold = config['critical_threshold']
        warning_threshold = config['warning_threshold']

        # Current stock level alerts
        if current_stock <= critical_threshold:
            alerts.append(StockAlert(
                type='current_critical',
                message=f"🚨 CRITICAL: Current stock ({current_stock:.1f}) is at or below critical threshold ({critical_threshold})",
                severity='critical',
                value=current_stock
            ))
        elif current_stock <= warning_threshold:
            alerts.append(StockAlert(
                type='current_warning',
                message=f"⚠️ WARNING: Current stock ({current_stock:.1f}) is below warning threshold ({warning_threshold})",
                severity='warning',
                value=current_stock
            ))

        # Trend-based alerts
        if len(df_product) >= 7:
            recent_trend = df_product['current_stock'].tail(7).iloc[-1] - df_product['current_stock'].tail(7).iloc[0]
            if recent_trend < -warning_threshold * 0.3:  # 30% of warning threshold
                alerts.append(StockAlert(
                    type='declining_trend',
                    message=f"📉 TREND ALERT: Stock declining rapidly (down {abs(recent_trend):.1f} units in 7 days)",
                    severity='warning',
                    value=recent_trend
                ))

        # Prediction-based alerts
        if predictions is not None and len(predictions) > 0:
            critical_days = predictions[predictions['predicted_stock'] <= critical_threshold]

            if not critical_days.empty:
                first_critical_day = critical_days.iloc[0]
                days_until = (first_critical_day['date'] - datetime.now()).days

                alerts.append(StockAlert(
                    type='predicted_critical',
                    message=f"⏰ PREDICTION: Stock will hit critical level in {days_until} days ({first_critical_day['date'].strftime('%Y-%m-%d')})",
                    severity='critical',
                    days_until=days_until
                ))

            # Check for stockout risk
            zero_stock_days = predictions[predictions['predicted_stock'] <= 0]
            if not zero_stock_days.empty:
                stockout_day = zero_stock_days.iloc[0]
                days_until = (stockout_day['date'] - datetime.now()).days

                alerts.append(StockAlert(
                    type='stockout_risk',
                    message=f"🛑 STOCKOUT RISK: Predicted stockout in {days_until} days ({stockout_day['date'].strftime('%Y-%m-%d')})",
                    severity='critical',
                    days_until=days_until
                ))

            # Optimal reorder point
            min_stock_day = predictions.loc[predictions['predicted_stock'].idxmin()]
            alerts.append(StockAlert(
                type='reorder_point',
                message=f"📋 REORDER: Minimum predicted stock {min_stock_day['predicted_stock']:.1f} units on {min_stock_day['date'].strftime('%Y-%m-%d')}",
                severity='info',
                value=min_stock_day['predicted_stock']
            ))

        return alerts


class ChartBuilder:
    """Enhanced chart building with multiple visualization options"""

    @staticmethod
    def create_comprehensive_chart(df_product: pd.DataFrame, predictions: Optional[pd.DataFrame],
                                   product_name: str, config: Dict) -> go.Figure:
        """Create comprehensive stock visualization"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                f'Stock Level Timeline - {product_name}',
                'Entry vs Exit Comparison',
                'Stock Velocity & Trends',
                'Volume Analysis',
                'Prediction Confidence',
                'Key Metrics Dashboard'
            ],
            specs=[
                [{"colspan": 2}, None],
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "indicator"}]
            ],
            vertical_spacing=0.08,
            row_heights=[0.5, 0.25, 0.25]
        )

        # Main stock level chart
        fig.add_trace(
            go.Scatter(
                x=df_product['date'],
                y=df_product['current_stock'],
                mode='lines+markers',
                name='Actual Stock',
                line=dict(color='#2E86C1', width=3),
                marker=dict(size=8, symbol='circle'),
                hovertemplate='<b>%{x}</b><br>Stock: %{y:.1f}<extra></extra>'
            ),
            row=1, col=1
        )

        # Add predictions
        if predictions is not None and len(predictions) > 0:
            fig.add_trace(
                go.Scatter(
                    x=predictions['date'],
                    y=predictions['predicted_stock'],
                    mode='lines+markers',
                    name='Predicted Stock',
                    line=dict(color='#E74C3C', width=3, dash='dash'),
                    marker=dict(size=6, symbol='diamond'),
                    hovertemplate='<b>%{x}</b><br>Predicted: %{y:.1f}<extra></extra>'
                ),
                row=1, col=1
            )

            # Confidence interval if available
            if hasattr(predictions, 'yhat_lower') or 'yhat_lower' in predictions.columns:
                fig.add_trace(
                    go.Scatter(
                        x=predictions['date'].tolist() + predictions['date'].tolist()[::-1],
                        y=predictions.get('yhat_upper', predictions['predicted_stock'] * 1.1).tolist() +
                          predictions.get('yhat_lower', predictions['predicted_stock'] * 0.9).tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(231, 76, 60, 0.1)',
                        line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo="skip",
                        showlegend=False,
                        name='Confidence Interval'
                    ),
                    row=1, col=1
                )

        # Threshold lines
        critical_threshold = config['critical_threshold']
        warning_threshold = config['warning_threshold']

        fig.add_hline(
            y=critical_threshold,
            line_dash="solid",
            line_color="#E74C3C",
            annotation_text=f"Critical ({critical_threshold})",
            row=1, col=1
        )

        fig.add_hline(
            y=warning_threshold,
            line_dash="dot",
            line_color="#F39C12",
            annotation_text=f"Warning ({warning_threshold})",
            row=1, col=1
        )

        # Entry/Exit comparison
        fig.add_trace(
            go.Scatter(
                x=df_product['date'],
                y=df_product['entry'],
                mode='lines',
                name='Daily Entry',
                line=dict(color='#27AE60', width=2),
                yaxis='y2'
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=df_product['date'],
                y=df_product['exit'],
                mode='lines',
                name='Daily Exit',
                line=dict(color='#E74C3C', width=2),
                yaxis='y2'
            ),
            row=2, col=1
        )

        # Volume analysis
        net_flow = df_product['entry'] - df_product['exit']
        colors = ['green' if x > 0 else 'red' for x in net_flow]

        fig.add_trace(
            go.Bar(
                x=df_product['date'],
                y=net_flow,
                name='Net Flow',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=2
        )

        # Stock velocity
        if 'stock_velocity' in df_product.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_product['date'],
                    y=df_product['stock_velocity'] * 100,  # Convert to percentage
                    mode='lines+markers',
                    name='Stock Velocity (%)',
                    line=dict(color='#8E44AD', width=2)
                ),
                row=3, col=1
            )

        # Key metrics indicator
        current_stock = df_product['current_stock'].iloc[-1]
        avg_daily_exit = df_product['exit'].mean()
        days_supply = current_stock / avg_daily_exit if avg_daily_exit > 0 else float('inf')

        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=days_supply,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Days of Supply"},
                gauge={
                    'axis': {'range': [None, 90]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 7], 'color': "lightgray"},
                        {'range': [7, 30], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 14
                    }
                }
            ),
            row=3, col=2
        )

        # Update layout
        fig.update_layout(
            height=900,
            showlegend=True,
            title=f"Comprehensive Stock Analysis - {product_name}",
            hovermode='x unified',
            template=config.get('chart_theme', 'plotly_white')
        )

        # Update axes
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Stock Quantity", row=1, col=1)
        fig.update_yaxes(title_text="Entry/Exit", row=2, col=1)
        fig.update_yaxes(title_text="Net Flow", row=2, col=2)
        fig.update_yaxes(title_text="Velocity %", row=3, col=1)

        return fig


def load_and_validate_data(uploaded_file) -> Optional[pd.DataFrame]:
    """Enhanced data loading with multiple format support"""
    try:
        file_extension = Path(uploaded_file.name).suffix.lower()

        if file_extension == '.csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(uploaded_file)
        elif file_extension == '.json':
            df = pd.read_json(uploaded_file)
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return None

        # Validate structure
        is_valid, errors = DataValidator.validate_data_structure(df)

        if not is_valid:
            st.error("Data validation failed:")
            for error in errors:
                st.error(f"• {error}")
            return None

        # Clean and preprocess
        df_cleaned = DataValidator.clean_and_preprocess(df)

        # Detect anomalies
        df_with_anomalies = DataValidator.detect_anomalies(df_cleaned)

        # Show data quality summary
        if len(df_cleaned) < len(df):
            st.warning(f"Removed {len(df) - len(df_cleaned)} rows with incomplete data")

        anomaly_count = df_with_anomalies['is_anomaly'].sum()
        if anomaly_count > 0:
            st.info(f"Detected {anomaly_count} potential anomalies in the data")

        return df_with_anomalies

    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None


def create_dashboard_metrics(df_product: pd.DataFrame, predictions: Optional[pd.DataFrame] = None,
                             config: Dict = None) -> None:
    """Create enhanced dashboard metrics"""
    if config is None:
        config = ConfigManager.get_default_config()

    current_stock = df_product['current_stock'].iloc[-1]
    avg_daily_exit = df_product['exit'].mean()
    avg_daily_entry = df_product['entry'].mean()

    # Calculate advanced metrics
    stock_turnover = df_product['exit'].sum() / df_product['current_stock'].mean() if df_product[
                                                                                          'current_stock'].mean() > 0 else 0
    days_supply = current_stock / avg_daily_exit if avg_daily_exit > 0 else float('inf')
    fill_rate = (df_product['current_stock'] > 0).mean() * 100

    # Recent trend (last 7 days)
    if len(df_product) >= 7:
        recent_change = df_product['current_stock'].iloc[-1] - df_product['current_stock'].iloc[-7]
        recent_change_pct = (recent_change / df_product['current_stock'].iloc[-7]) * 100 if \
        df_product['current_stock'].iloc[-7] != 0 else 0
    else:
        recent_change = 0
        recent_change_pct = 0

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Current Stock",
            f"{current_stock:.0f}",
            delta=f"{recent_change:+.0f} ({recent_change_pct:+.1f}%)"
        )

    with col2:
        days_supply_display = f"{days_supply:.1f}" if days_supply != float('inf') else "∞"
        color = "normal"
        if days_supply < 7:
            color = "inverse"
        elif days_supply < 14:
            color = "off"

        st.metric(
            "Days Supply",
            days_supply_display,
            help="Days of inventory remaining at current consumption rate"
        )

    with col3:
        st.metric(
            "Avg Daily Flow",
            f"+{avg_daily_entry:.1f}/-{avg_daily_exit:.1f}",
            delta=f"Net: {avg_daily_entry - avg_daily_exit:+.1f}"
        )

    with col4:
        st.metric(
            "Turnover Rate",
            f"{stock_turnover:.2f}",
            help="How many times inventory is sold/used per period"
        )

    with col5:
        st.metric(
            "Availability",
            f"{fill_rate:.1f}%",
            help="Percentage of days with stock available"
        )

    # Prediction metrics if available
    if predictions is not None and len(predictions) > 0:
        st.divider()
        st.subheader("30-Day Forecast Metrics")

        pred_final = predictions['predicted_stock'].iloc[-1]
        pred_change = pred_final - current_stock
        pred_change_pct = (pred_change / current_stock) * 100 if current_stock != 0 else 0

        # Risk metrics
        critical_threshold = config['critical_threshold']
        critical_days = predictions[predictions['predicted_stock'] <= critical_threshold]
        days_to_critical = len(critical_days) if not critical_days.empty else None

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Predicted Stock (30d)",
                f"{pred_final:.0f}",
                delta=f"{pred_change:+.0f} ({pred_change_pct:+.1f}%)"
            )

        with col2:
            if days_to_critical:
                st.metric(
                    "Days to Critical",
                    f"{days_to_critical}",
                    delta="Critical risk"
                )
            else:
                st.metric("Risk Level", "Low", delta="Safe levels predicted")

        with col3:
            min_predicted = predictions['predicted_stock'].min()
            st.metric(
                "Minimum Predicted",
                f"{min_predicted:.0f}",
                help="Lowest predicted stock level in forecast period"
            )

        with col4:
            pred_volatility = predictions['predicted_stock'].std()
            st.metric(
                "Forecast Volatility",
                f"{pred_volatility:.1f}",
                help="Standard deviation of predictions"
            )


import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from typing import Optional


def create_advanced_analytics(df_product: pd.DataFrame) -> None:
    """
    Create comprehensive advanced analytics section for inventory management.

    Args:
        df_product: DataFrame with columns ['date', 'entry', 'exit', 'current_stock', 'net_change']
    """
    st.subheader("📊 Advanced Analytics")

    # Validate required columns
    required_cols = ['date', 'entry', 'exit', 'current_stock']
    if not all(col in df_product.columns for col in required_cols):
        st.error(f"Missing required columns. Expected: {required_cols}")
        return

    if df_product.empty:
        st.warning("No data available for analysis")
        return

    # Create tabs for different analytics sections
    tab1, tab2, tab3, tab4 = st.tabs(["🗓️ Seasonality", "🔗 Correlations", "⚠️ Anomalies", "📈 Performance"])

    with tab1:
        _create_seasonality_analysis(df_product)

    with tab2:
        _create_correlation_analysis(df_product)

    with tab3:
        _create_anomaly_analysis(df_product)

    with tab4:
        _create_performance_metrics(df_product)


def _create_seasonality_analysis(df_product: pd.DataFrame) -> None:
    """Analyze seasonal patterns in inventory data."""
    st.write("### Seasonal Pattern Analysis")

    if len(df_product) < 30:
        st.info("⏳ Need at least 30 data points to analyze seasonality patterns effectively")
        return

    # Ensure date column is datetime
    df_product = df_product.copy()
    df_product['date'] = pd.to_datetime(df_product['date'])

    # Add time-based features
    df_product['day_of_week'] = df_product['date'].dt.day_name()
    df_product['month'] = df_product['date'].dt.month_name()
    df_product['hour'] = df_product['date'].dt.hour if df_product['date'].dt.hour.nunique() > 1 else None

    # Weekly patterns analysis
    st.write("#### Weekly Patterns")
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_pattern = df_product.groupby('day_of_week').agg({
        'entry': ['mean', 'std'],
        'exit': ['mean', 'std'],
        'current_stock': 'mean'
    }).round(2)

    weekly_pattern.columns = ['_'.join(col).strip() for col in weekly_pattern.columns.values]
    weekly_pattern = weekly_pattern.reindex(day_order, fill_value=0)

    # Create weekly pattern visualization
    fig_weekly = go.Figure()
    fig_weekly.add_trace(go.Bar(
        x=weekly_pattern.index,
        y=weekly_pattern['entry_mean'],
        name='Average Entry',
        marker_color='lightgreen',
        error_y=dict(type='data', array=weekly_pattern['entry_std'])
    ))
    fig_weekly.add_trace(go.Bar(
        x=weekly_pattern.index,
        y=weekly_pattern['exit_mean'],
        name='Average Exit',
        marker_color='lightcoral',
        error_y=dict(type='data', array=weekly_pattern['exit_std'])
    ))

    fig_weekly.update_layout(
        title="Average Entry/Exit by Day of Week (with Standard Deviation)",
        xaxis_title="Day of Week",
        yaxis_title="Quantity",
        barmode='group',
        showlegend=True
    )
    st.plotly_chart(fig_weekly, use_container_width=True)

    # Monthly patterns (if data spans multiple months)
    if df_product['date'].dt.month.nunique() > 1:
        st.write("#### Monthly Patterns")
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']

        monthly_pattern = df_product.groupby('month').agg({
            'entry': 'mean',
            'exit': 'mean',
            'current_stock': 'mean',
            'net_change': 'mean'
        }).round(2)

        # Reorder by calendar months
        available_months = [month for month in month_order if month in monthly_pattern.index]
        monthly_pattern = monthly_pattern.reindex(available_months)

        fig_monthly = go.Figure()
        fig_monthly.add_trace(go.Scatter(
            x=monthly_pattern.index,
            y=monthly_pattern['entry'],
            mode='lines+markers',
            name='Average Entry',
            line=dict(color='green')
        ))
        fig_monthly.add_trace(go.Scatter(
            x=monthly_pattern.index,
            y=monthly_pattern['exit'],
            mode='lines+markers',
            name='Average Exit',
            line=dict(color='red')
        ))

        fig_monthly.update_layout(
            title="Average Entry/Exit by Month",
            xaxis_title="Month",
            yaxis_title="Quantity",
            showlegend=True
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

        # Show monthly insights
        best_entry_month = monthly_pattern['entry'].idxmax()
        best_exit_month = monthly_pattern['exit'].idxmax()
        st.info(f"📈 Highest entries typically occur in **{best_entry_month}**")
        st.info(f"📉 Highest exits typically occur in **{best_exit_month}**")


def _create_correlation_analysis(df_product: pd.DataFrame) -> None:
    """Analyze correlations between different inventory metrics."""
    st.write("### Correlation Analysis")

    # Select numeric columns for correlation
    numeric_cols = ['current_stock', 'entry', 'exit']
    if 'net_change' in df_product.columns:
        numeric_cols.append('net_change')
    if 'stock_velocity' in df_product.columns:
        numeric_cols.append('stock_velocity')

    # Calculate correlation matrix
    corr_data = df_product[numeric_cols].corr()

    # Create correlation heatmap
    fig_corr = px.imshow(
        corr_data,
        text_auto='.3f',
        aspect="auto",
        title="Inventory Metrics Correlation Matrix",
        color_continuous_scale="RdBu",
        range_color=[-1, 1]
    )
    fig_corr.update_layout(
        width=600,
        height=500
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # Extract and display key insights
    st.write("#### Key Correlation Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Stock Level Relationships:**")
        entry_stock_corr = corr_data.loc['entry', 'current_stock']
        exit_stock_corr = corr_data.loc['exit', 'current_stock']

        st.metric("Entry ↔ Stock Level", f"{entry_stock_corr:.3f}")
        st.metric("Exit ↔ Stock Level", f"{exit_stock_corr:.3f}")

        # Interpretation
        if abs(entry_stock_corr) > 0.5:
            direction = "positive" if entry_stock_corr > 0 else "negative"
            st.success(f"🔗 Strong {direction} correlation between entries and stock levels")

        if abs(exit_stock_corr) > 0.5:
            direction = "positive" if exit_stock_corr > 0 else "negative"
            st.success(f"🔗 Strong {direction} correlation between exits and stock levels")

    with col2:
        st.write("**Activity Relationships:**")
        if 'net_change' in numeric_cols:
            entry_exit_corr = corr_data.loc['entry', 'exit']
            st.metric("Entry ↔ Exit", f"{entry_exit_corr:.3f}")

            if entry_exit_corr > 0.3:
                st.info("📊 Entries and exits tend to occur together")
            elif entry_exit_corr < -0.3:
                st.info("📊 Entries and exits are inversely related")

        # Find strongest correlation (excluding self-correlations)
        corr_copy = corr_data.copy()
        np.fill_diagonal(corr_copy.values, 0)
        max_corr_idx = np.unravel_index(np.argmax(np.abs(corr_copy.values)), corr_copy.shape)
        strongest_pair = (corr_copy.index[max_corr_idx[0]], corr_copy.columns[max_corr_idx[1]])
        strongest_value = corr_copy.iloc[max_corr_idx[0], max_corr_idx[1]]

        st.write("**Strongest Correlation:**")
        st.write(f"{strongest_pair[0]} ↔ {strongest_pair[1]}: **{strongest_value:.3f}**")


def _create_anomaly_analysis(df_product: pd.DataFrame) -> None:
    """Detect and visualize anomalies in inventory data."""
    st.write("### Anomaly Detection")

    # Check if anomalies are pre-computed
    if 'is_anomaly' in df_product.columns and 'anomaly_score' in df_product.columns:
        anomalies = df_product[df_product['is_anomaly']]
        _display_precomputed_anomalies(df_product, anomalies)
    else:
        # Compute anomalies using statistical methods
        _compute_and_display_anomalies(df_product)


def _display_precomputed_anomalies(df_product: pd.DataFrame, anomalies: pd.DataFrame) -> None:
    """Display pre-computed anomalies."""
    if len(anomalies) > 0:
        st.write(
            f"🔍 Found **{len(anomalies)}** anomalous data points ({len(anomalies) / len(df_product) * 100:.1f}% of data)")

        # Visualize anomalies
        fig_anomaly = go.Figure()

        # Normal data points
        normal_data = df_product[~df_product['is_anomaly']]
        fig_anomaly.add_trace(go.Scatter(
            x=normal_data['date'],
            y=normal_data['current_stock'],
            mode='lines+markers',
            name='Normal Data',
            marker=dict(size=4, color='blue', opacity=0.6),
            line=dict(color='blue', width=1)
        ))

        # Anomalous data points
        fig_anomaly.add_trace(go.Scatter(
            x=anomalies['date'],
            y=anomalies['current_stock'],
            mode='markers',
            name='Anomalies',
            marker=dict(size=10, color='red', symbol='x-thin', line=dict(width=2))
        ))

        fig_anomaly.update_layout(
            title="Stock Level Anomalies Detection",
            xaxis_title="Date",
            yaxis_title="Stock Level",
            hovermode='x unified'
        )

        st.plotly_chart(fig_anomaly, use_container_width=True)

        # Anomaly summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Anomalies", len(anomalies))
        with col2:
            st.metric("Avg Anomaly Score", f"{anomalies['anomaly_score'].mean():.2f}")
        with col3:
            st.metric("Max Anomaly Score", f"{anomalies['anomaly_score'].max():.2f}")

        # Detailed anomaly table
        with st.expander("🔍 Detailed Anomaly Information"):
            display_cols = ['date', 'current_stock', 'entry', 'exit', 'anomaly_score']
            available_cols = [col for col in display_cols if col in anomalies.columns]
            st.dataframe(
                anomalies[available_cols].sort_values('anomaly_score', ascending=False),
                use_container_width=True
            )
    else:
        st.success("✅ No significant anomalies detected in the data")


def _compute_and_display_anomalies(df_product: pd.DataFrame) -> None:
    """Compute anomalies using statistical methods."""
    st.write("Computing anomalies using statistical analysis...")

    # Use Z-score method for anomaly detection
    df_temp = df_product.copy()

    # Calculate Z-scores for stock levels
    stock_mean = df_temp['current_stock'].mean()
    stock_std = df_temp['current_stock'].std()

    if stock_std == 0:
        st.info("No variability in stock levels - no anomalies can be detected")
        return

    df_temp['z_score'] = np.abs((df_temp['current_stock'] - stock_mean) / stock_std)

    # Define anomalies as points with |Z-score| > 2.5
    threshold = st.slider("Anomaly Detection Threshold (Z-score)", 1.5, 4.0, 2.5, 0.1)
    df_temp['is_anomaly'] = df_temp['z_score'] > threshold

    anomalies = df_temp[df_temp['is_anomaly']]

    if len(anomalies) > 0:
        st.write(f"🔍 Detected **{len(anomalies)}** anomalies using Z-score > {threshold}")

        # Visualize
        fig = go.Figure()

        # Normal points
        normal_data = df_temp[~df_temp['is_anomaly']]
        fig.add_trace(go.Scatter(
            x=normal_data['date'],
            y=normal_data['current_stock'],
            mode='lines+markers',
            name='Normal',
            marker=dict(size=4, color='blue')
        ))

        # Anomalies
        fig.add_trace(go.Scatter(
            x=anomalies['date'],
            y=anomalies['current_stock'],
            mode='markers',
            name='Anomalies',
            marker=dict(size=12, color='red', symbol='x')
        ))

        fig.update_layout(
            title=f"Anomaly Detection (Z-score > {threshold})",
            xaxis_title="Date",
            yaxis_title="Stock Level"
        )

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Anomaly Details"):
            st.dataframe(
                anomalies[['date', 'current_stock', 'entry', 'exit', 'z_score']].sort_values('z_score',
                                                                                             ascending=False),
                use_container_width=True
            )
    else:
        st.success(f"✅ No anomalies detected with Z-score threshold > {threshold}")


def _create_performance_metrics(df_product: pd.DataFrame) -> None:
    """Calculate and display inventory performance metrics."""
    st.write("### Inventory Performance Metrics")

    # Get configuration values
    critical_threshold = st.session_state.get('config_critical_threshold', 20)
    target_days_supply = st.session_state.get('target_days_supply', 14)

    # Allow user to adjust critical threshold
    with st.expander("Configuration"):
        critical_threshold = st.number_input("Critical Stock Threshold", value=critical_threshold, min_value=0)
        target_days_supply = st.number_input("Target Days Supply", value=target_days_supply, min_value=1)

    # Calculate metrics
    current_stock = df_product['current_stock'].iloc[-1] if not df_product.empty else 0
    avg_daily_exit = df_product['exit'].mean() if not df_product.empty else 0
    max_stock = df_product['current_stock'].max()
    avg_stock = df_product['current_stock'].mean()

    # Service Level (% of time above critical threshold)
    service_level = (df_product['current_stock'] > critical_threshold).mean() * 100

    # Inventory Utilization
    utilization = (avg_stock / max_stock) * 100 if max_stock > 0 else 0

    # Demand Variability (Coefficient of Variation)
    exit_cv = df_product['exit'].std() / df_product['exit'].mean() if df_product['exit'].mean() > 0 else 0

    # Days Supply
    days_supply = current_stock / avg_daily_exit if avg_daily_exit > 0 else float('inf')

    # Stockout Risk
    stockout_risk = (df_product['current_stock'] == 0).mean() * 100

    # Create metrics DataFrame
    metrics_data = [
        {
            'Metric': 'Service Level (%)',
            'Current': f"{service_level:.1f}%",
            'Target': '≥ 95%',
            'Status': '✅' if service_level >= 95 else '❌',
            'Description': f'% of time stock > {critical_threshold} units'
        },
        {
            'Metric': 'Inventory Utilization (%)',
            'Current': f"{utilization:.1f}%",
            'Target': '60-85%',
            'Status': '✅' if 60 <= utilization <= 85 else '❌',
            'Description': 'Average stock / Maximum stock'
        },
        {
            'Metric': 'Demand Variability (CV)',
            'Current': f"{exit_cv:.2f}",
            'Target': '≤ 0.50',
            'Status': '✅' if exit_cv <= 0.5 else '❌',
            'Description': 'Standard deviation / Mean of exits'
        },
        {
            'Metric': 'Current Days Supply',
            'Current': f"{days_supply:.1f}" if days_supply != float('inf') else "∞",
            'Target': f'{target_days_supply} days',
            'Status': '✅' if 7 <= days_supply <= 30 else '❌',
            'Description': 'Current stock / Average daily exits'
        },
        {
            'Metric': 'Stockout Risk (%)',
            'Current': f"{stockout_risk:.1f}%",
            'Target': '< 5%',
            'Status': '✅' if stockout_risk < 5 else '❌',
            'Description': '% of time with zero stock'
        }
    ]

    metrics_df = pd.DataFrame(metrics_data)

    # Display metrics in a more visual way
    col1, col2 = st.columns([3, 2])

    with col1:
        st.dataframe(
            metrics_df[['Metric', 'Current', 'Target', 'Status']],
            use_container_width=True,
            hide_index=True
        )

    with col2:
        # Performance summary
        passed_metrics = sum(1 for _, row in metrics_df.iterrows() if row['Status'] == '✅')
        total_metrics = len(metrics_df)
        performance_score = (passed_metrics / total_metrics) * 100

        st.metric("Overall Performance Score", f"{performance_score:.0f}%")

        if performance_score >= 80:
            st.success("Excellent inventory performance!")
        elif performance_score >= 60:
            st.warning("⚠️ Good performance with room for improvement")
        else:
            st.error("🔧 Performance needs attention")

    # Detailed explanations
    with st.expander("Metric Explanations"):
        for _, row in metrics_df.iterrows():
            st.write(f"**{row['Metric']}:** {row['Description']}")


# Example usage function
def example_usage():
    """Example of how to use the advanced analytics function."""
    st.title("Inventory Analytics Dashboard")

    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)

    sample_data = pd.DataFrame({
        'date': dates,
        'entry': np.random.poisson(15, 100),
        'exit': np.random.poisson(12, 100),
        'current_stock': None
    })

    # Calculate current stock
    sample_data['current_stock'] = 100  # Starting stock
    for i in range(1, len(sample_data)):
        sample_data.loc[i, 'current_stock'] = (
                sample_data.loc[i - 1, 'current_stock'] +
                sample_data.loc[i, 'entry'] -
                sample_data.loc[i, 'exit']
        )

    sample_data['net_change'] = sample_data['entry'] - sample_data['exit']

    # Run the analytics
    create_advanced_analytics(sample_data)


# Uncomment the line below to test the function
# example_usage()

def export_analysis_report(df_product: pd.DataFrame, predictions: Optional[pd.DataFrame],
                           alerts: List[StockAlert], ai_analysis: str = "") -> str:
    """Generate comprehensive analysis report for export"""
    report = f"""
# Stock Analysis Report
**Product:** {df_product['product'].iloc[0]}
**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Analysis Period:** {df_product['date'].min().strftime('%Y-%m-%d')} to {df_product['date'].max().strftime('%Y-%m-%d')}

## Executive Summary
- **Current Stock Level:** {df_product['current_stock'].iloc[-1]:.1f} units
- **Average Daily Consumption:** {df_product['exit'].mean():.1f} units
- **Days of Supply:** {df_product['current_stock'].iloc[-1] / df_product['exit'].mean():.1f} days

## Key Alerts
"""

    for alert in alerts[:5]:  # Top 5 alerts
        report += f"- {alert.message}\n"

    if predictions is not None:
        report += f"""
## 30-Day Forecast
- **Predicted Stock Level:** {predictions['predicted_stock'].iloc[-1]:.1f} units
- **Expected Change:** {predictions['predicted_stock'].iloc[-1] - df_product['current_stock'].iloc[-1]:+.1f} units
- **Minimum Predicted Level:** {predictions['predicted_stock'].min():.1f} units
"""

    if ai_analysis:
        report += f"""
## AI Analysis Insights
{ai_analysis}
"""

    report += f"""
## Data Summary
- **Total Records:** {len(df_product)}
- **Total Entries:** {df_product['entry'].sum():.1f} units
- **Total Exits:** {df_product['exit'].sum():.1f} units
- **Stock Volatility:** {df_product['current_stock'].std():.2f}

---
*Report generated by AI-Enhanced Stock Tracking System*
"""

    return report


def show_prediction_flowchart():
    """Display the AI prediction pipeline flowchart using Mermaid diagram"""

    # Add section header with context
    st.subheader("🔄 AI Prediction Workflow")
    st.write("This diagram shows how your inventory data flows through our AI prediction system:")

    # Mermaid diagram with improved styling
    mermaid_code = """
    flowchart TD
        A[📁 Upload Inventory Data] --> B[🧹 Clean & Validate Data]
        B --> C{Enough Data for Prediction?}
        C -- Yes --> D{Prophet Available?}
        D -- Yes --> E[🔮 Use Prophet for Time-Series Forecast]
        D -- No --> F[📉 Use Regression Model]
        C -- No --> G[❌ Show Error: Not Enough Data]
        E --> H[📈 Generate 30-Day Forecast]
        F --> H
        H --> I{Any Alerts Triggered?}
        I -- Yes --> J[🚨 Create Alerts<br/>Critical, Stockout, Reorder]
        I -- No --> K[✅ No Risk Detected]
        H --> L[🧠 Generate AI Insights<br/>LLaMA 3.1 via Ollama]
        L --> M[🗂️ Output Dashboard & Report]
        J --> M
        K --> M

        classDef inputNode fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
        classDef processNode fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
        classDef decisionNode fill:#fff3e0,stroke:#f57c00,stroke-width:2px
        classDef outputNode fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
        classDef errorNode fill:#ffebee,stroke:#d32f2f,stroke-width:2px

        class A inputNode
        class B,E,F,H,L processNode
        class C,D,I decisionNode
        class M,K outputNode
        class G,J errorNode
    """

    st.markdown(f"```mermaid\n{mermaid_code}\n```")

    # Add explanatory note
    st.info(
        "💡 **Tip**: The system automatically selects the best prediction model based on your data characteristics and available libraries. LLaMA 3.1 analysis requires Ollama running locally.")

def main():
    st.set_page_config(
        page_title="AI-Enhanced Stock Tracking & Prediction",
        #page_icon="X",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    if 'config' not in st.session_state:
        st.session_state['config'] = ConfigManager.get_default_config()

    st.title("AI-Enhanced Stock Tracking & Prediction System")
    st.markdown("Advanced inventory analytics with AI-powered insights and predictive modeling")

    # Initialize components
    llama = LlamaAnalyzer()
    config = ConfigManager.load_config()

    # Check AI availability
    llama_available = llama.is_available()

    # Enhanced sidebar
    with st.sidebar:
        st.header("Configuration")

        # AI Settings
        st.subheader("AI Assistant")
        if llama_available:
            st.success("🟢 Llama 3.1: Connected")
            enable_ai = st.checkbox("Enable AI Analysis", value=True)
            if enable_ai:
                ai_mode = st.selectbox(
                    "Analysis Mode",
                    ["Comprehensive", "Quick Insights", "Risk Assessment", "Custom"]
                )
        else:
            st.error("🔴 Llama 3.1: Not Available")
            st.info("Install Ollama and run: `ollama run llama3.1`")
            enable_ai = False

        st.divider()

        # Threshold Settings
        st.subheader(" Thresholds")
        critical_threshold = st.slider(
            "Critical Level", 1, 100, config['critical_threshold'],
            help="Alert when stock falls below this level"
        )
        warning_threshold = st.slider(
            "Warning Level", critical_threshold + 1, 200, config['warning_threshold'],
            help="Warning when stock falls below this level"
        )

        # Prediction Settings
        st.subheader("Predictions")
        prediction_days = st.slider("Forecast Days", 7, 90, config['prediction_days'])

        # Chart Settings
        st.subheader("Display")
        chart_theme = st.selectbox(
            "Chart Theme",
            ["plotly_white", "plotly_dark", "ggplot2", "seaborn"],
            index=0
        )

        # Update config
        config.update({
            'critical_threshold': critical_threshold,
            'warning_threshold': warning_threshold,
            'prediction_days': prediction_days,
            'chart_theme': chart_theme
        })

    # File upload section
    st.header("📁 Data Upload")

    col1, col2 = st.columns([3, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload your inventory data",
            type=['csv', 'xlsx', 'json'],
            help="Supported formats: CSV, Excel, JSON"
        )


    if uploaded_file is not None:
        # Load and validate data
        with st.spinner("Loading and validating data..."):
            df = load_and_validate_data(uploaded_file)

        if df is not None:
            st.success(f"✅ Data loaded successfully! {len(df)} records, {df['product'].nunique()} products")

            # Data overview
            st.header("Data Overview")
            create_dashboard_metrics(df, config=config)

            # Product selection
            st.header("Product Analysis")
            products = sorted(df['product'].unique())
            selected_product = st.selectbox("Select Product for Detailed Analysis", products)

            # Filter data for selected product
            df_product = df[df['product'] == selected_product].sort_values('date')

            if len(df_product) < 3:
                st.error(f"Product '{selected_product}' needs at least 3 data points for analysis")
                return

            # Generate predictions
            try:
                with st.spinner("Generating predictive models..."):
                    prediction_result = PredictionEngine.predict_stock(df_product, prediction_days)

                st.success(f"✅ Predictions generated using {prediction_result.model_type}")

                if prediction_result.accuracy_score:
                    st.info(f"Model accuracy: {prediction_result.accuracy_score:.2%}")

            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                prediction_result = None

            # Generate alerts
            alerts = AlertSystem.generate_alerts(
                df_product,
                prediction_result.predictions if prediction_result else None,
                config
            )

            # Display alerts
            if alerts:
                st.header("🚨 Alerts & Notifications")

                critical_alerts = [a for a in alerts if a.severity == 'critical']
                warning_alerts = [a for a in alerts if a.severity == 'warning']
                info_alerts = [a for a in alerts if a.severity == 'info']

                if critical_alerts:
                    st.subheader("🚨 Critical Alerts")
                    for alert in critical_alerts:
                        st.error(alert.message)

                if warning_alerts:
                    st.subheader("⚠️ Warnings")
                    for alert in warning_alerts:
                        st.warning(alert.message)

                if info_alerts:
                    with st.expander("ℹ️ Additional Information"):
                        for alert in info_alerts:
                            st.info(alert.message)

            # Main visualization
            st.header("Interactive Analytics")

            if prediction_result:
                fig = ChartBuilder.create_comprehensive_chart(
                    df_product,
                    prediction_result.predictions,
                    selected_product,
                    config
                )
                st.plotly_chart(fig, use_container_width=True)

            # Product-specific metrics
            create_dashboard_metrics(
                df_product,
                prediction_result.predictions if prediction_result else None,
                config
            )

            # AI Analysis Section
            if enable_ai and llama_available:
                st.header("Analysis & Insights")

                col1, col2, col3 = st.columns([2, 2, 1])

                with col1:
                    if st.button("Generate AI Analysis", type="primary"):
                        with st.spinner("AI analyzing your inventory data..."):
                            ai_analysis = llama.analyze_comprehensive(
                                df_product,
                                prediction_result.predictions if prediction_result else None,
                                alerts
                            )
                            st.session_state['ai_analysis'] = ai_analysis

                with col2:
                    custom_question = st.text_input(
                        "Ask AI a Custom Question",
                        placeholder="e.g., What's the optimal reorder point?"
                    )

                    if st.button("Ask AI") and custom_question:
                        with st.spinner("AI processing your question..."):
                            context = f"""
Product: {selected_product}
Current Stock: {df_product['current_stock'].iloc[-1]:.1f}
Recent Trend: {df_product['current_stock'].iloc[-7:].mean():.1f} (7-day avg)
"""
                            custom_response = llama.generate_analysis(custom_question, context)
                            st.session_state['custom_ai_response'] = custom_response

                with col3:
                    if st.button("Clear Analysis"):
                        if 'ai_analysis' in st.session_state:
                            del st.session_state['ai_analysis']
                        if 'custom_ai_response' in st.session_state:
                            del st.session_state['custom_ai_response']

                # Display AI responses
                if 'ai_analysis' in st.session_state:
                    st.subheader("Comprehensive AI Analysis")
                    st.markdown(st.session_state['ai_analysis'])

                if 'custom_ai_response' in st.session_state:
                    st.subheader("AI Response to Your Question")
                    st.markdown(st.session_state['custom_ai_response'])

            # Advanced analytics
            create_advanced_analytics(df_product)

            # Export functionality
            st.header("Export Analysis")

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("Export Report"):
                    ai_text = st.session_state.get('ai_analysis', '')
                    report = export_analysis_report(
                        df_product,
                        prediction_result.predictions if prediction_result else None,
                        alerts,
                        ai_text
                    )

                    st.download_button(
                        label="Download Report",
                        data=report,
                        file_name=f"stock_analysis_{selected_product}_{datetime.now().strftime('%Y%m%d')}.md",
                        mime="text/markdown"
                    )

            with col2:
                if prediction_result and st.button("Export Predictions"):
                    csv_data = prediction_result.predictions.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=f"predictions_{selected_product}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )

            with col3:
                if st.button("Export Raw Data"):
                    csv_data = df_product.to_csv(index=False)
                    st.download_button(
                        label="ownload CSV",
                        data=csv_data,
                        file_name=f"raw_data_{selected_product}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )

            # Raw data view
            with st.expander("🔍 Raw Data Inspection"):
                st.dataframe(df_product, use_container_width=True)

    else:
        # Welcome screen and setup instructions
        st.header("AI-Enhanced Stock Tracking")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("✨ Key Features")
            st.markdown("""
            - **🤖 AI-Powered Analysis**: Get intelligent insights with Llama 3.1
            - **🔮 Advanced Predictions**: Prophet & machine learning models
            - **🚨 Smart Alerts**: Proactive inventory monitoring
            - **📊 Interactive Dashboards**: Rich visualizations and analytics
            - **📈 Performance Metrics**: KPIs and efficiency tracking
            - **🔍 Anomaly Detection**: Identify unusual patterns
            - **📤 Export Capabilities**: Reports and data export
            """)

        with col2:
            st.subheader("Setup Instructions")

            if not llama_available:
                st.markdown("""
                **Enable AI Features:**
                1. Install [Ollama](https://ollama.ai)
                2. Run: `ollama run llama3.1`
                3. Refresh this page

                **Requirements:**
                - 8GB+ RAM recommended
                - ~4.7GB storage for Llama 3.1
                """)
            else:
                st.success("🟢 AI features are ready!")

            st.markdown("""
            **Data Format:**
            Upload CSV/Excel with columns:
            - `date`: Date (YYYY-MM-DD)
            - `product`: Product name
            - `entry`: Units added
            - `exit`: Units consumed
            - `current_stock`: Current inventory
            """)

    # Footer
    st.divider()
    st.markdown(
        "Built with ❤️ with BABTICH EL Habib using Streamlit, Plotly, and AI • "
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )


if __name__ == "__main__":
    main()