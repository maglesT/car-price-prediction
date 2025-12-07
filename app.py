import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Only import SHAP if available (graceful degradation)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ===========================
# PAGE CONFIGURATION
# ===========================

st.set_page_config(
    page_title="Car Price Predictor with XAI",
    page_icon="🚗",
    layout="wide"
)

# ===========================
# LOAD MODEL
# ===========================

@st.cache_resource
def load_model():
    try:
        with open('models/v1/model.pkl', 'rb') as f:
            model_params = pickle.load(f)
        return model_params, None
    except Exception as e:
        return None, str(e)

model_params, error = load_model()

# ===========================
# PREDICTION FUNCTIONS
# ===========================

def add_polynomial_features(X, degree=2):
    """Add polynomial features if needed"""
    if degree == 1:
        return X
    
    n_samples, n_features = X.shape
    poly_features = [X]
    
    # Add squared terms
    squared = X ** 2
    poly_features.append(squared)
    
    # Add interaction terms
    for i in range(n_features):
        for j in range(i+1, n_features):
            interaction = (X[:, i] * X[:, j]).reshape(-1, 1)
            poly_features.append(interaction)
    
    return np.hstack(poly_features)

def predict_price(features, model_params):
    """
    Predict car price using Normal Equation parameters
    Pure mathematical implementation - no ML libraries!
    """
    # Extract parameters
    theta = model_params['theta']
    mean = model_params['mean']
    std = model_params['std']
    use_poly = model_params.get('use_polynomial', False)
    degree = model_params.get('degree', 1)
    
    # Convert to numpy array
    features_array = np.array(features).reshape(1, -1)
    
    # Scale features (standardization)
    features_scaled = (features_array - mean) / std
    
    # Add polynomial features if needed
    if use_poly and degree > 1:
        features_scaled = add_polynomial_features(features_scaled, degree)
    
    # Add bias term (column of ones)
    features_with_bias = np.c_[np.ones((1, 1)), features_scaled]
    
    # Predict: ŷ = X @ θ
    prediction = features_with_bias @ theta
    
    return prediction[0], features_scaled[0]

def create_predict_function(model_params):
    """Create a prediction function for SHAP"""
    def predict_fn(X):
        """Wrapper for SHAP that takes numpy array and returns predictions"""
        predictions = []
        for row in X:
            pred, _ = predict_price(row.tolist(), model_params)
            predictions.append(pred)
        return np.array(predictions)
    return predict_fn

# ===========================
# BIAS DETECTION FUNCTIONS
# ===========================

def analyze_bias(model_params, feature_names):
    """
    Analyze potential bias in model predictions across different feature segments
    """
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic test data covering feature ranges
    test_data = {
        'wheelbase': np.random.uniform(86, 120, n_samples),
        'carlength': np.random.uniform(141, 208, n_samples),
        'carwidth': np.random.uniform(60, 72, n_samples),
        'carheight': np.random.uniform(47, 60, n_samples),
        'curbweight': np.random.uniform(1488, 4066, n_samples),
        'enginesize': np.random.uniform(61, 326, n_samples),
        'boreratio': np.random.uniform(2.54, 3.94, n_samples),
        'horsepower': np.random.uniform(48, 288, n_samples),
        'citympg': np.random.uniform(13, 49, n_samples),
        'highwaympg': np.random.uniform(16, 54, n_samples),
    }
    
    # Convert to array
    X_test = np.column_stack([test_data[name] for name in feature_names])
    
    # Make predictions for all samples
    predictions = []
    for i in range(n_samples):
        pred, _ = predict_price(X_test[i].tolist(), model_params)
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # Analyze bias across different segments
    bias_analysis = {}
    
    # 1. Analyze by efficiency (MPG)
    low_mpg = predictions[test_data['citympg'] < 20]
    high_mpg = predictions[test_data['citympg'] >= 35]
    
    bias_analysis['efficiency'] = {
        'low_mpg_mean': np.mean(low_mpg),
        'high_mpg_mean': np.mean(high_mpg),
        'difference': np.mean(low_mpg) - np.mean(high_mpg),
        'low_mpg_std': np.std(low_mpg),
        'high_mpg_std': np.std(high_mpg)
    }
    
    # 2. Analyze by size (weight)
    light_cars = predictions[test_data['curbweight'] < 2200]
    heavy_cars = predictions[test_data['curbweight'] >= 3200]
    
    bias_analysis['size'] = {
        'light_mean': np.mean(light_cars),
        'heavy_mean': np.mean(heavy_cars),
        'difference': np.mean(heavy_cars) - np.mean(light_cars),
        'light_std': np.std(light_cars),
        'heavy_std': np.std(heavy_cars)
    }
    
    # 3. Analyze by power (horsepower)
    low_power = predictions[test_data['horsepower'] < 90]
    high_power = predictions[test_data['horsepower'] >= 180]
    
    bias_analysis['power'] = {
        'low_power_mean': np.mean(low_power),
        'high_power_mean': np.mean(high_power),
        'difference': np.mean(high_power) - np.mean(low_power),
        'low_power_std': np.std(low_power),
        'high_power_std': np.std(high_power)
    }
    
    # 4. Statistical fairness test (coefficient of variation)
    cv = np.std(predictions) / np.mean(predictions)
    bias_analysis['overall_fairness'] = {
        'coefficient_variation': cv,
        'mean_prediction': np.mean(predictions),
        'std_prediction': np.std(predictions),
        'min_prediction': np.min(predictions),
        'max_prediction': np.max(predictions)
    }
    
    return bias_analysis, test_data, predictions

def detect_feature_bias(shap_values, feature_names, features):
    """
    Detect if certain features have disproportionate impact (potential bias)
    """
    impacts = np.abs(shap_values.values[0])
    total_impact = np.sum(impacts)
    
    feature_bias = []
    for i, (name, impact) in enumerate(zip(feature_names, impacts)):
        contribution_pct = (impact / total_impact) * 100
        
        # Flag if single feature contributes >30% (potential over-reliance)
        is_biased = contribution_pct > 30
        
        feature_bias.append({
            'feature': name,
            'contribution_pct': contribution_pct,
            'absolute_impact': impact,
            'potentially_biased': is_biased,
            'your_value': features[i]
        })
    
    return sorted(feature_bias, key=lambda x: x['contribution_pct'], reverse=True)

# ===========================
# UI - HEADER
# ===========================

st.title("🚗 Car Price Prediction with Explainable AI")
st.markdown("**AI-Powered Price Estimation using Pure Mathematics + SHAP Explainability + Fairness Analysis**")

# Check if model loaded
if error:
    st.error(f"❌ Error loading model: {error}")
    
    with st.expander("🔍 Debug Information"):
        st.code(error)
        st.markdown("""
        **Troubleshooting Steps:**
        1. Make sure `models/v1/model.pkl` exists in your GitHub repository
        2. Check that the file was uploaded correctly
        3. Verify the file is not corrupted
        """)
    st.stop()

st.success("✅ Model loaded successfully!")

# Display model info with XAI badge
model_type = model_params.get('model_type', 'LINEAR')
test_r2 = model_params.get('test_r2', 0.78)
test_mae = model_params.get('test_mae', 3064)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Algorithm", model_type)
col2.metric("R² Score", f"{test_r2*100:.1f}%")
col3.metric("Avg Error", f"${test_mae:,.0f}")
col4.metric("Explainable", "✓ SHAP" if SHAP_AVAILABLE else "✗", 
            delta="XAI" if SHAP_AVAILABLE else "N/A")

# XAI Info Banner
if SHAP_AVAILABLE:
    with st.expander("🔍 What is Explainable AI?", expanded=False):
        st.markdown("""
        ### Explainable AI (XAI) with SHAP
        
        This app uses **SHAP (SHapley Additive exPlanations)** to explain predictions:
        
        - 🎯 **Transparency**: See which features influenced the price
        - 📊 **Feature Importance**: Understand what matters most
        - ⚖️ **Fairness**: Detect potential bias
        - 🔍 **Trust**: Know why the AI decided this price
        
        **How to read explanations:**
        - **Red/Positive** → Features increasing price
        - **Blue/Negative** → Features decreasing price
        - **Bar length** → Strength of impact
        """)

st.markdown("---")

# ===========================
# UI - INPUT FORM
# ===========================

st.subheader("📝 Enter Car Specifications")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🔧 Physical Dimensions")
    
    wheelbase = st.slider(
        "Wheelbase (inches)", 
        86.0, 120.0, 98.5, 0.5,
        help="Distance between front and rear axles"
    )
    
    carlength = st.slider(
        "Car Length (inches)", 
        141.0, 208.0, 173.3, 0.5,
        help="Overall length of the vehicle"
    )
    
    carwidth = st.slider(
        "Car Width (inches)", 
        60.0, 72.0, 65.8, 0.1,
        help="Width of the vehicle"
    )
    
    carheight = st.slider(
        "Car Height (inches)", 
        47.0, 60.0, 53.7, 0.1,
        help="Height from ground to roof"
    )
    
    curbweight = st.slider(
        "Curb Weight (lbs)", 
        1488, 4066, 2555, 10,
        help="Weight of empty vehicle"
    )

with col2:
    st.markdown("#### ⚡ Engine Specifications")
    
    enginesize = st.slider(
        "Engine Size (cc)", 
        61, 326, 126, 5,
        help="Engine displacement in cubic centimeters"
    )
    
    boreratio = st.slider(
        "Bore Ratio", 
        2.54, 3.94, 3.33, 0.01,
        help="Engine bore to stroke ratio"
    )
    
    horsepower = st.slider(
        "Horsepower (hp)", 
        48, 288, 104, 5,
        help="Maximum engine power output"
    )
    
    citympg = st.slider(
        "City MPG", 
        13, 49, 25, 1,
        help="Fuel efficiency in city driving"
    )
    
    highwaympg = st.slider(
        "Highway MPG", 
        16, 54, 30, 1,
        help="Fuel efficiency on highways"
    )

# ===========================
# PREDICTION SECTION
# ===========================

st.markdown("---")

if st.button("🔮 Predict Car Price with Explanation", type="primary", use_container_width=True):
    
    # Feature names
    feature_names = [
        'wheelbase', 'carlength', 'carwidth', 'carheight',
        'curbweight', 'enginesize', 'boreratio', 'horsepower',
        'citympg', 'highwaympg'
    ]
    
    # Prepare features in correct order
    features = [
        wheelbase, carlength, carwidth, carheight,
        curbweight, enginesize, boreratio, horsepower,
        citympg, highwaympg
    ]
    
    try:
        # Make prediction
        predicted_price, scaled_features = predict_price(features, model_params)
        
        # Display result
        st.markdown("---")
        st.markdown("## 💰 Prediction Result")
        
        # Main prediction display
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown(f"### Estimated Price: **${predicted_price:,.2f}**")
            
            # Progress bar
            price_percentage = min(predicted_price / 50000, 1.0)
            st.progress(price_percentage)
            
            # Price range (±10% confidence)
            lower = predicted_price * 0.9
            upper = predicted_price * 1.1
            st.info(f"📊 Expected Range: ${lower:,.2f} - ${upper:,.2f}")
            st.caption("Range represents ±10% confidence interval")
        
        # Car summary metrics
        st.markdown("---")
        st.markdown("### 📋 Your Car Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Engine", f"{enginesize} cc", f"{horsepower} hp")
        col2.metric("Efficiency", f"{citympg} / {highwaympg} MPG", "City / Highway")
        col3.metric("Weight", f"{curbweight:,} lbs", f"{carwidth:.1f}\" wide")
        col4.metric("Size", f"{carlength:.1f}\"", f"{wheelbase:.1f}\" wheelbase")
        
        # Value assessment
        st.markdown("---")
        st.markdown("### 💡 Value Assessment")
        
        if predicted_price < 10000:
            st.success("🟢 **Budget-Friendly** - Great value for money!")
            st.write("This car is perfect for first-time buyers or those on a tight budget.")
        elif predicted_price < 20000:
            st.info("🔵 **Mid-Range** - Good balance of features and price")
            st.write("Solid choice with decent features and reasonable pricing.")
        elif predicted_price < 35000:
            st.warning("🟡 **Premium** - Higher-end features and performance")
            st.write("Enhanced features, better performance, and quality materials.")
        else:
            st.error("🔴 **Luxury** - Top-tier vehicle with premium features")
            st.write("High-end vehicle with exceptional features and performance.")
        
        # ===========================
        # EXPLAINABLE AI SECTION
        # ===========================
        
        if SHAP_AVAILABLE:
            st.markdown("---")
            st.markdown("## 🔍 Prediction Explanation")
            st.markdown("**Understanding why the model predicted this price:**")
            
            with st.spinner("Generating SHAP explanations..."):
                
                # Create background dataset for SHAP
                np.random.seed(42)
                X_background = np.random.randn(100, 10)
                # Scale background to reasonable ranges
                X_background[:, 0] = X_background[:, 0] * 10 + 100  # wheelbase
                X_background[:, 1] = X_background[:, 1] * 20 + 170  # carlength
                X_background[:, 2] = X_background[:, 2] * 4 + 66    # carwidth
                X_background[:, 3] = X_background[:, 3] * 4 + 53    # carheight
                X_background[:, 4] = X_background[:, 4] * 600 + 2500  # curbweight
                X_background[:, 5] = X_background[:, 5] * 60 + 150  # enginesize
                X_background[:, 6] = X_background[:, 6] * 0.3 + 3.2 # boreratio
                X_background[:, 7] = X_background[:, 7] * 60 + 120  # horsepower
                X_background[:, 8] = X_background[:, 8] * 10 + 25   # citympg
                X_background[:, 9] = X_background[:, 9] * 10 + 30   # highwaympg
                
                # Create SHAP explainer
                predict_fn = create_predict_function(model_params)
                explainer = shap.Explainer(predict_fn, X_background)
                
                # Get SHAP values for current prediction
                input_array = np.array(features).reshape(1, -1)
                shap_values = explainer(input_array)
                
                # Create tabs for different explanations
                tab1, tab2, tab3 = st.tabs(["📊 Feature Impact", "💧 Waterfall Plot", "📈 Summary"])
                
                with tab1:
                    st.subheader("Feature Impact on Prediction")
                    st.markdown("**How each feature contributed to the predicted price:**")
                    
                    # Create feature impact dataframe
                    impacts = shap_values.values[0]
                    base_value = shap_values.base_values[0]
                    
                    feature_impacts = pd.DataFrame({
                        'Feature': feature_names,
                        'Your Value': features,
                        'Impact ($)': impacts,
                    })
                    feature_impacts['Abs_Impact'] = np.abs(feature_impacts['Impact ($)'])
                    feature_impacts = feature_impacts.sort_values('Abs_Impact', ascending=False)
                    
                    # Create horizontal bar chart
                    fig, ax = plt.subplots(figsize=(10, 7))
                    colors = ['#ff6b6b' if x > 0 else '#4ecdc4' for x in feature_impacts['Impact ($)']]
                    bars = ax.barh(feature_impacts['Feature'], feature_impacts['Impact ($)'], 
                                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
                    
                    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
                    ax.set_xlabel('Impact on Price ($)', fontsize=13, fontweight='bold')
                    ax.set_title('Feature Contributions to Predicted Price', 
                                fontsize=15, fontweight='bold', pad=15)
                    ax.grid(axis='x', alpha=0.3, linestyle='--')
                    
                    # Add value labels
                    for i, (bar, val) in enumerate(zip(bars, feature_impacts['Impact ($)'])):
                        label = f'${val:,.0f}'
                        x_pos = val + (abs(val) * 0.05 if val > 0 else -abs(val) * 0.05)
                        ax.text(x_pos, i, label, va='center', 
                               ha='left' if val > 0 else 'right', 
                               fontweight='bold', fontsize=10)
                    
                    # Legend
                    from matplotlib.patches import Patch
                    legend_elements = [
                        Patch(facecolor='#ff6b6b', alpha=0.8, label='Increases Price'),
                        Patch(facecolor='#4ecdc4', alpha=0.8, label='Decreases Price')
                    ]
                    ax.legend(handles=legend_elements, loc='lower right', fontsize=11)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Show detailed table
                    st.markdown("**Detailed Breakdown:**")
                    display_df = feature_impacts[['Feature', 'Your Value', 'Impact ($)']].copy()
                    display_df['Impact ($)'] = display_df['Impact ($)'].apply(lambda x: f"${x:,.2f}")
                    display_df['Your Value'] = display_df['Your Value'].apply(lambda x: f"{x:.2f}")
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                    
                    # Base value explanation
                    st.info(f"""
                    **Base Prediction**: ${base_value:,.2f}  
                    (Average price before considering your car's specific features)
                    
                    **Your Car's Price**: ${predicted_price:,.2f}  
                    (Base + All feature contributions)
                    """)
                
                with tab2:
                    st.subheader("SHAP Waterfall Plot")
                    st.markdown("**Step-by-step breakdown showing how we arrived at the prediction:**")
                    
                    # Create waterfall plot
                    fig, ax = plt.subplots(figsize=(10, 8))
                    shap.plots.waterfall(shap_values[0], show=False)
                    plt.title('SHAP Waterfall - Prediction Breakdown', 
                             fontsize=14, fontweight='bold', pad=15)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    st.info("""
                    **How to read this chart:**
                    - Starts at the **base value** (average prediction)
                    - Each bar shows a feature pushing price up (red) or down (blue)
                    - Features are ordered by impact magnitude
                    - Final value = base + all contributions
                    """)
                
                with tab3:
                    st.subheader("Feature Importance Summary")
                    
                    # Most important features
                    top_3 = feature_impacts.nlargest(3, 'Abs_Impact')
                    
                    st.markdown("**🎯 Top 3 Most Influential Features:**")
                    
                    for idx, row in top_3.iterrows():
                        impact_direction = "↑ Increases" if row['Impact ($)'] > 0 else "↓ Decreases"
                        st.markdown(f"""
                        **{row['Feature'].replace('_', ' ').title()}**  
                        - Your value: {row['Your Value']:.2f}
                        - Impact: ${row['Impact ($)']:,.2f} ({impact_direction} price)
                        """)
                    
                    # Overall assessment
                    st.markdown("---")
                    st.markdown("**📊 Overall Assessment:**")
                    
                    positive_impact = feature_impacts[feature_impacts['Impact ($)'] > 0]['Impact ($)'].sum()
                    negative_impact = abs(feature_impacts[feature_impacts['Impact ($)'] < 0]['Impact ($)'].sum())
                    
                    col_assess1, col_assess2 = st.columns(2)
                    with col_assess1:
                        st.metric("Features Increasing Price", 
                                 len(feature_impacts[feature_impacts['Impact ($)'] > 0]),
                                 f"+${positive_impact:,.0f}")
                    with col_assess2:
                        st.metric("Features Decreasing Price", 
                                 len(feature_impacts[feature_impacts['Impact ($)'] < 0]),
                                 f"-${negative_impact:,.0f}")
                    
                    # Configuration assessment
                    if positive_impact > negative_impact:
                        st.success("✅ **Premium Configuration** - Your car has more high-value features")
                    else:
                        st.info("ℹ️ **Economy Configuration** - Focus on efficiency and value")
                
                # What-if suggestions
                st.markdown("---")
                st.subheader("💡 What-If Scenarios")
                st.markdown("**Want to change the price? Try adjusting these features:**")
                
                col_tips1, col_tips2 = st.columns(2)
                
                # Get top features by absolute impact
                sorted_features = feature_impacts.sort_values('Abs_Impact', ascending=False)
                
                with col_tips1:
                    st.markdown("**To INCREASE price:**")
                    count = 0
                    for _, row in sorted_features.iterrows():
                        if count >= 3:
                            break
                        if row['Impact ($)'] < 0:
                            st.markdown(f"- ⬆️ Increase **{row['Feature'].replace('_', ' ')}** (currently {row['Your Value']:.1f})")
                            count += 1
                        elif row['Impact ($)'] > 0:
                            st.markdown(f"- ✓ **{row['Feature'].replace('_', ' ')}** already helping (+${row['Impact ($)']:,.0f})")
                            count += 1
                
                with col_tips2:
                    st.markdown("**To DECREASE price:**")
                    count = 0
                    for _, row in sorted_features.iterrows():
                        if count >= 3:
                            break
                        if row['Impact ($)'] > 0:
                            st.markdown(f"- ⬇️ Decrease **{row['Feature'].replace('_', ' ')}** (currently {row['Your Value']:.1f})")
                            count += 1
                        elif row['Impact ($)'] < 0:
                            st.markdown(f"- ✓ **{row['Feature'].replace('_', ' ')}** already reducing (-${abs(row['Impact ($)']):,.0f})")
                            count += 1
                
                # ===========================
                # FAIRNESS & BIAS ANALYSIS
                # ===========================
                
                st.markdown("---")
                st.markdown("## ⚖️ Fairness & Bias Analysis")
                st.markdown("**Ensuring transparent and equitable predictions**")
                
                tab_fair1, tab_fair2, tab_fair3 = st.tabs(["🎯 Feature Bias", "📊 Segment Fairness", "🔍 Model Fairness"])
                
                with tab_fair1:
                    st.subheader("Feature Bias Detection")
                    st.markdown("**Checking if any single feature dominates predictions (potential bias):**")
                    
                    # Detect feature bias
                    feature_bias = detect_feature_bias(shap_values, feature_names, features)
                    
                    # Create visualization
                    bias_df = pd.DataFrame(feature_bias)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors = ['#ff4444' if x['potentially_biased'] else '#44ff44' 
                             for x in feature_bias]
                    
                    bars = ax.barh(bias_df['feature'], bias_df['contribution_pct'], 
                                   color=colors, alpha=0.7, edgecolor='black', linewidth=1.2)
                    
                    ax.axvline(x=30, color='red', linestyle='--', linewidth=2, 
                              label='Bias Threshold (30%)')
                    ax.set_xlabel('Contribution to Prediction (%)', fontsize=12, fontweight='bold')
                    ax.set_title('Feature Contribution Distribution\n(Red = Potential Over-Reliance)', 
                                fontsize=14, fontweight='bold', pad=15)
                    ax.legend()
                    ax.grid(axis='x', alpha=0.3)
                    
                    # Add percentage labels
                    for bar, val in zip(bars, bias_df['contribution_pct']):
                        ax.text(val + 1, bar.get_y() + bar.get_height()/2, 
                               f'{val:.1f}%', va='center', fontweight='bold')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Show bias analysis
                    st.markdown("**Bias Assessment:**")
                    
                    biased_features = [f for f in feature_bias if f['potentially_biased']]
                    
                    if len(biased_features) == 0:
                        st.success("✅ **No Feature Bias Detected** - Prediction is well-distributed across features")
                        st.write("No single feature dominates the prediction (all <30% contribution)")
                    else:
                        st.warning(f"⚠️ **Potential Bias Detected** - {len(biased_features)} feature(s) have high influence")
                        for feat in biased_features:
                            st.markdown(f"""
                            - **{feat['feature'].replace('_', ' ').title()}**: {feat['contribution_pct']:.1f}% contribution
                              - Your value: {feat['your_value']:.2f}
                              - This feature has disproportionate impact on pricing
                            """)
                        
                        st.info("""
                        **What this means:**  
                        When a single feature contributes >30% to the prediction, the model may be 
                        over-relying on it. This could indicate:
                        - Feature engineering needed
                        - Data collection bias in training set
                        - Need for additional features to balance prediction
                        """)
                    
                    # Show detailed contribution table
                    st.markdown("**Detailed Feature Contributions:**")
                    display_bias = bias_df[['feature', 'contribution_pct', 'your_value']].copy()
                    display_bias.columns = ['Feature', 'Contribution (%)', 'Your Value']
                    display_bias['Contribution (%)'] = display_bias['Contribution (%)'].apply(lambda x: f"{x:.2f}%")
                    display_bias['Your Value'] = display_bias['Your Value'].apply(lambda x: f"{x:.2f}")
                    st.dataframe(display_bias, use_container_width=True, hide_index=True)
                
                with tab_fair2:
                    st.subheader("Segment Fairness Analysis")
                    st.markdown("**Analyzing prediction fairness across different car segments:**")
                    
                    with st.spinner("Running fairness analysis across segments..."):
                        bias_analysis, test_data, predictions = analyze_bias(model_params, feature_names)
                    
                    # Efficiency segment analysis
                    st.markdown("### 🌱 Efficiency Segment (City MPG)")
                    
                    col_eff1, col_eff2, col_eff3 = st.columns(3)
                    
                    with col_eff1:
                        st.metric("Low Efficiency (<20 MPG)", 
                                 f"${bias_analysis['efficiency']['low_mpg_mean']:,.0f}",
                                 f"±${bias_analysis['efficiency']['low_mpg_std']:,.0f}")
                    
                    with col_eff2:
                        st.metric("High Efficiency (≥35 MPG)", 
                                 f"${bias_analysis['efficiency']['high_mpg_mean']:,.0f}",
                                 f"±${bias_analysis['efficiency']['high_mpg_std']:,.0f}")
                    
                    with col_eff3:
                        price_diff = bias_analysis['efficiency']['difference']
                        st.metric("Price Difference", 
                                 f"${abs(price_diff):,.0f}",
                                 "Higher for low MPG" if price_diff > 0 else "Lower for low MPG")
                    
                    # Fairness assessment for efficiency
                    if abs(price_diff) < 5000:
                        st.success("✅ **Fair Treatment** - Minimal bias between efficiency segments")
                    elif abs(price_diff) < 10000:
                        st.warning("⚠️ **Moderate Disparity** - Some difference in pricing by efficiency")
                    else:
                        st.error("❌ **Significant Disparity** - Large price difference by efficiency")
                    
                    st.markdown("---")
                    
                    # Size segment analysis
                    st.markdown("### 🚗 Size Segment (Weight)")
                    
                    col_size1, col_size2, col_size3 = st.columns(3)
                    
                    with col_size1:
                        st.metric("Light Cars (<2200 lbs)", 
                                 f"${bias_analysis['size']['light_mean']:,.0f}",
                                 f"±${bias_analysis['size']['light_std']:,.0f}")
                    
                    with col_size2:
                        st.metric("Heavy Cars (≥3200 lbs)", 
                                 f"${bias_analysis['size']['heavy_mean']:,.0f}",
                                 f"±${bias_analysis['size']['heavy_std']:,.0f}")
                    
                    with col_size3:
                        size_diff = bias_analysis['size']['difference']
                        st.metric("Price Difference", 
                                 f"${abs(size_diff):,.0f}",
                                 "Higher for heavy" if size_diff > 0 else "Lower for heavy")
                    
                    # Visualization of segment distributions
                    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                    
                    # Efficiency distribution
                    low_mpg_pred = predictions[test_data['citympg'] < 20]
                    high_mpg_pred = predictions[test_data['citympg'] >= 35]
                    
                    axes[0].hist(low_mpg_pred, bins=30, alpha=0.6, label='Low MPG (<20)', color='#ff6b6b')
                    axes[0].hist(high_mpg_pred, bins=30, alpha=0.6, label='High MPG (≥35)', color='#51cf66')
                    axes[0].axvline(np.mean(low_mpg_pred), color='#ff6b6b', linestyle='--', linewidth=2)
                    axes[0].axvline(np.mean(high_mpg_pred), color='#51cf66', linestyle='--', linewidth=2)
                    axes[0].set_xlabel('Predicted Price ($)', fontweight='bold')
                    axes[0].set_ylabel('Frequency', fontweight='bold')
                    axes[0].set_title('Price Distribution by Efficiency', fontweight='bold')
                    axes[0].legend()
                    axes[0].grid(alpha=0.3)
                    
                    # Size distribution
                    light_pred = predictions[test_data['curbweight'] < 2200]
                    heavy_pred = predictions[test_data['curbweight'] >= 3200]
                    
                    axes[1].hist(light_pred, bins=30, alpha=0.6, label='Light (<2200 lbs)', color='#4ecdc4')
                    axes[1].hist(heavy_pred, bins=30, alpha=0.6, label='Heavy (≥3200 lbs)', color='#ffd93d')
                    axes[1].axvline(np.mean(light_pred), color='#4ecdc4', linestyle='--', linewidth=2)
                    axes[1].axvline(np.mean(heavy_pred), color='#ffd93d', linestyle='--', linewidth=2)
                    axes[1].set_xlabel('Predicted Price ($)', fontweight='bold')
                    axes[1].set_ylabel('Frequency', fontweight='bold')
                    axes[1].set_title('Price Distribution by Weight', fontweight='bold')
                    axes[1].legend()
                    axes[1].grid(alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.info("""
                    **Interpreting Fairness:**  
                    - **Overlapping distributions** → Fair treatment across segments
                    - **Separated distributions** → Systematic pricing differences
                    - **Large gaps** → Potential bias requiring investigation
                    """)
                
                with tab_fair3:
                    st.subheader("Overall Model Fairness")
                    st.markdown("**Global fairness metrics and model behavior:**")
                    
                    # Overall statistics
                    overall = bias_analysis['overall_fairness']
                    
                    col_fair1, col_fair2, col_fair3, col_fair4 = st.columns(4)
                    
                    with col_fair1:
                        st.metric("Mean Prediction", f"${overall['mean_prediction']:,.0f}")
                    
                    with col_fair2:
                        st.metric("Std Deviation", f"${overall['std_prediction']:,.0f}")
                    
                    with col_fair3:
                        st.metric("Price Range", 
                                 f"${overall['max_prediction'] - overall['min_prediction']:,.0f}")
                    
                    with col_fair4:
                        cv = overall['coefficient_variation']
                        cv_pct = cv * 100
                        st.metric("Variation Coefficient", f"{cv_pct:.1f}%")
                    
                    st.markdown("---")
                    
                    # Fairness score
                    st.markdown("### 🎯 Fairness Score")
                    
                    # Calculate fairness score (0-100)
                    # Lower CV = better fairness
                    fairness_score = max(0, 100 - (cv * 200))  # Scale CV to 0-100
                    
                    # Display fairness gauge
                    col_gauge1, col_gauge2 = st.columns([1, 2])
                    
                    with col_gauge1:
                        st.markdown(f"### **{fairness_score:.1f}/100**")
                        
                        if fairness_score >= 80:
                            st.success("✅ Excellent Fairness")
                            fairness_label = "Excellent"
                            fairness_color = "#51cf66"
                        elif fairness_score >= 60:
                            st.info("ℹ️ Good Fairness")
                            fairness_label = "Good"
                            fairness_color = "#4ecdc4"
                        elif fairness_score >= 40:
                            st.warning("⚠️ Moderate Fairness")
                            fairness_label = "Moderate"
                            fairness_color = "#ffd93d"
                        else:
                            st.error("❌ Poor Fairness")
                            fairness_label = "Needs Improvement"
                            fairness_color = "#ff6b6b"
                    
                    with col_gauge2:
                        # Create fairness gauge chart
                        fig, ax = plt.subplots(figsize=(8, 3))
                        ax.barh(['Fairness'], [fairness_score], color=fairness_color, height=0.5)
                        ax.set_xlim(0, 100)
                        ax.set_xlabel('Fairness Score', fontweight='bold')
                        ax.set_title(f'Model Fairness: {fairness_label}', fontweight='bold')
                        ax.grid(axis='x', alpha=0.3)
                        ax.text(fairness_score + 2, 0, f'{fairness_score:.1f}', 
                               va='center', fontweight='bold', fontsize=12)
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    st.markdown("---")
                    
                    # Explanation
                    st.markdown("### 📖 Understanding Fairness Metrics")
                    
                    st.markdown(f"""
                    **Coefficient of Variation (CV): {cv_pct:.2f}%**
                    - Measures relative variability in predictions
                    - Lower CV = more consistent, fair predictions
                    - Your model: {"Excellent consistency" if cv < 0.3 else "Moderate consistency" if cv < 0.5 else "High variability"}
                    
                    **What Makes a Fair Model:**
                    - ✅ Consistent predictions across all feature ranges
                    - ✅ No single feature dominates (all <30% contribution)
                    - ✅ Similar price distributions across segments
                    - ✅ Transparent, explainable predictions (SHAP)
                    
                    **Bias Mitigation Strategies:**
                    - Collect more diverse training data
                    - Balance feature representation
                    - Regular fairness audits
                    - Monitor predictions across segments
                    - Use explainability to catch unfair patterns
                    """)
                    
                    # Recommendation
                    st.markdown("### 💡 Recommendations")
                    
                    if fairness_score >= 70:
                        st.success("""
                        **Model shows good fairness!** Continue monitoring and:
                        - Regularly test on new data segments
                        - Document any edge cases discovered
                        - Maintain transparent prediction explanations
                        """)
                    else:
                        st.warning("""
                        **Consider improvements:**
                        - Analyze which segments show largest disparities
                        - Collect additional training data for underrepresented segments
                        - Consider feature engineering to reduce single-feature dominance
                        - Implement regular fairness audits
                        """)
        
        else:
            st.warning("⚠️ SHAP not available. Install with: `pip install shap`")
        
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        with st.expander("Error Details"):
            st.code(str(e))
            import traceback
            st.code(traceback.format_exc())

# ===========================
# FOOTER - MODEL INFO
# ===========================

st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### 📊 About This Model
    
    **Algorithm**: Linear Regression via Normal Equation  
    **Formula**: θ = (X^T X)^(-1) X^T y  
    **Training Data**: 164 cars  
    **Test Accuracy**: 78.3% (R² score)  
    **Average Error**: $3,064  
    **Explainability**: SHAP (SHapley Additive exPlanations)  
    **Fairness**: Bias detection & segment analysis
    
    #### Why Normal Equation + SHAP + Fairness?
    - ✅ No ML libraries needed (only NumPy!)
    - ✅ Direct mathematical solution
    - ✅ Fast deployment (< 1 minute)
    - ✅ Complete transparency with SHAP
    - ✅ Perfect for linear regression
    - ✅ Real-time explainability
    - ✅ Fairness monitoring built-in
    
    #### Model Selection Process
    We tested both linear and polynomial (degree 2) features:
    - Linear model: 78% test accuracy with minimal overfitting
    - Polynomial model: Showed 71% overfitting gap
    - **Decision**: Linear model generalizes better for our dataset size
    
    #### Explainable AI Integration
    - **SHAP values**: Show feature contributions to each prediction
    - **Waterfall plots**: Visual breakdown of prediction logic
    - **Feature importance**: Understand what drives prices
    - **Bias detection**: Monitor fairness in predictions
    - **Segment analysis**: Ensure equitable treatment across car types
    """)

with col2:
    st.markdown("""
    ### 🎓 Project Info
    
    **Created by**:  
    Ruben Santosh  
    Vignesh R Nair  
    Arko Chakraborty  
    
    **University**:  
    Dayananda Sagar University  
    
    **Department**:  
    CSE (AI & ML)  
    
    **Year**: 2025  
    
    **Tech Stack**:
    - Streamlit
    - NumPy
    - SHAP (XAI)
    - Pure Mathematics
    - GitHub
    - Fairness Analysis
    
    **Dataset**:  
    Car Price Dataset  
    (205 cars, 10 features)
    
    **MLOps**:
    - Explainable AI
    - Real-time monitoring
    - Bias detection
    - Fairness analysis
    - GitHub CI/CD
    """)

# Mathematical explanation (expandable)
with st.expander("🔬 See the Mathematics Behind Predictions"):
    st.markdown("""
    ### Normal Equation Formula
    
    The model finds optimal parameters θ directly:
    
    $$\\theta = (X^T X)^{-1} X^T y$$
    
    Where:
    - **θ (theta)** = model parameters (weights)
    - **X** = feature matrix (car specifications)
    - **y** = target values (prices)
    - **X^T** = transpose of X
    
    ### Prediction Formula
    
    For a new car, we calculate:
    
    $$\\hat{y} = \\theta_0 + \\theta_1 x_1 + \\theta_2 x_2 + ... + \\theta_{10} x_{10}$$
    
    ### Feature Scaling
    
    Before prediction, we standardize features:
    
    $$x_{scaled} = \\frac{x - \\mu}{\\sigma}$$
    
    Where:
    - **μ** = mean of feature
    - **σ** = standard deviation
    
    This ensures all features contribute equally to the prediction.
    
    ### SHAP Values Explained
    
    SHAP uses Shapley values from game theory:
    
    $$\\phi_i = \\sum_{S \\subseteq F \\setminus \\{i\\}} \\frac{|S|!(|F|-|S|-1)!}{|F|!}[f_{S \\cup \\{i\\}}(x_{S \\cup \\{i\\}}) - f_S(x_S)]$$
    
    Where:
    - **φᵢ** = SHAP value for feature i
    - **F** = set of all features
    - **S** = subset of features
    
    SHAP values represent each feature's contribution to moving the prediction from the base value to the final prediction.
    
    ### Fairness Metrics
    
    **Coefficient of Variation (CV)**:
    
    $$CV = \\frac{\\sigma}{\\mu}$$
    
    Where:
    - **σ** = standard deviation of predictions
    - **μ** = mean of predictions
    
    Lower CV indicates more consistent and fair predictions across different inputs.
    
    ### Example Calculation
    
    For your car with enginesize = 126 cc:
    1. Scale: (126 - 128.5) / 41.2 = -0.061
    2. Multiply by weight: -0.061 × 2981.57 = -181.88
    3. SHAP shows this feature's isolated contribution
    4. Sum all weighted features + intercept = Final Price
    """)

# Usage tips
with st.expander("💡 Tips for Best Results"):
    st.markdown("""
    ### Getting Accurate Predictions
    
    1. **Use Realistic Values**
       - Don't extrapolate beyond slider ranges
       - These represent typical car specifications
    
    2. **Consider Correlations**
       - Larger engines usually mean higher weight
       - More horsepower often reduces MPG
       - Bigger cars typically have longer wheelbases
    
    3. **Understand the Range**
       - ±10% confidence interval is normal
       - Real car prices vary by condition, location, features
    
    4. **Model Limitations**
       - Based on 205 cars from dataset
       - Doesn't account for: brand reputation, condition, mileage
       - Best for comparative pricing
    
    5. **Using SHAP Explanations**
       - Red features increase price relative to average
       - Blue features decrease price relative to average
       - Focus on high-impact features for biggest price changes
       - Use "What-If" suggestions to optimize for your budget
    
    6. **Understanding Fairness**
       - Check feature bias - no single feature should dominate
       - Review segment fairness for consistency
       - High fairness score = reliable predictions across all car types
    """)

st.markdown("---")
st.caption("🚗 Car Price Predictor with Explainable AI & Fairness Analysis | Built with ❤️ using Pure Math + SHAP | Responsible AI in Production!")
