import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import time

# Set professional layout configuration
st.set_page_config(
    page_title="Enterprise NIDS Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom injection for sleek, modern alert states
st.markdown("""
    <style>
    .metric-card-safe {
        background-color: #0e2f1d;
        border-left: 5px solid #2ecc71;
        padding: 15px;
        border-radius: 5px;
        color: #2ecc71;
    }
    .metric-card-danger {
        background-color: #3a161a;
        border-left: 5px solid #e74c3c;
        padding: 15px;
        border-radius: 5px;
        color: #e74c3c;
    }
    .badge-text {
        font-size: 24px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    return (
        joblib.load("encoders.pkl"),
        joblib.load("scaler.pkl"),
        joblib.load("feature_columns.pkl"),
        joblib.load("random_forest_model.pkl"),
        tf.keras.models.load_model("cnn_model.keras")
    )

encoders, scaler, feature_columns, rf_model, cnn_model = load_assets()
clean_features = [col for col in feature_columns if col != "connection"]
target_encoder = encoders["connection"]
class_labels = list(target_encoder.classes_)

# --- SIDEBAR OVERVIEW ---
with st.sidebar:
    st.image("https://img.icons8.com/nolan/128/shield.png", width=80)
    st.title("IDS Engine Overview")
    st.write("---")
    st.info("💡 **System Status:** Operational\n\n🎯 **Engines Active:** Random Forest + 1D-CNN")
    st.write("---")
    st.markdown("### Active Telemetry Stream")
    
    # Generate live telemetry plot
    np.random.seed(int(time.time()) % 1000)
    simulated_traffic = pd.DataFrame(
        np.random.randint(10, 80, size=(20, 2)),
        columns=['TCP Packets/s', 'UDP Packets/s']
    )
    st.line_chart(simulated_traffic, height=180)

# --- MAIN CONTROLS INTERFACE ---
st.title("🛡️ Enterprise Network Intrusion Detection System")
st.markdown("Adjust the categorized real-time connection signature configurations below to analyze packet behavior profiles.")

# Dynamic Slicing: Split the exact clean features array evenly across 3 tabs
chunks = np.array_split(clean_features, 3)
basic_features_group = list(chunks[0])
content_features_group = list(chunks[1])
traffic_features_group = list(chunks[2])

user_inputs = {}

tab_basic, tab_content, tab_traffic = st.tabs(["🌐 Basic Connection Metrics", "📁 Content Analytics", "📈 Statistical Traffic Profiles"])

with tab_basic:
    st.subheader("Core Packet Headers")
    col1, col2, col3 = st.columns(3)
    for idx, col in enumerate(basic_features_group):
        target_col = [col1, col2, col3][idx % 3]
        with target_col:
            if col in encoders:
                user_inputs[col] = st.selectbox(f"**{col}**", list(encoders[col].classes_), key=f"basic_{col}")
            else:
                user_inputs[col] = st.number_input(
                    label=f"**{col}**", 
                    value=0.0, 
                    step=1.0 if col in ['src_bytes', 'dst_bytes'] else 0.1, 
                    key=f"basic_num_{col}"
                )

with tab_content:
    st.subheader("Payload Flag & Authentication Contexts")
    col1, col2, col3 = st.columns(3)
    for idx, col in enumerate(content_features_group):
        target_col = [col1, col2, col3][idx % 3]
        with target_col:
            if col in encoders:
                user_inputs[col] = st.selectbox(f"**{col}**", list(encoders[col].classes_), key=f"content_{col}")
            else:
                user_inputs[col] = st.number_input(
                    label=f"**{col}**", 
                    value=0.0, 
                    step=1.0, 
                    key=f"content_num_{col}"
                )

with tab_traffic:
    st.subheader("Window-Based Traffic Statistics")
    col1, col2, col3 = st.columns(3)
    for idx, col in enumerate(traffic_features_group):
        target_col = [col1, col2, col3][idx % 3]
        with target_col:
            if col in encoders:
                user_inputs[col] = st.selectbox(f"**{col}**", list(encoders[col].classes_), key=f"traffic_{col}")
            else:
                user_inputs[col] = st.number_input(
                    label=f"**{col}**", 
                    value=0.0, 
                    step=0.01 if 'rate' in col else 1.0, 
                    key=f"traffic_num_{col}"
                )

st.write("---")

# --- ANALYSIS ENGINE PIPELINE EXECUTION ---
if st.button("🚀 Analyze Active Network Packet Signature", type="primary", use_container_width=True):
    with st.spinner("Decoding threat payload metrics through parallel heuristics engines..."):
        start_time = time.time()
        
        input_df = pd.DataFrame([user_inputs]).reindex(columns=clean_features)
        
        # Apply transformation matrices
        for col in input_df.columns:
            if col in encoders:
                current_val = str(input_df[col].iloc[0])
                input_df[col] = encoders[col].transform([current_val])[0]
                
        scaled_data = scaler.transform(input_df)
        
        # Compute Inference Profiles
        rf_pred = rf_model.predict(scaled_data)[0]
        cnn_probs = cnn_model.predict(np.expand_dims(scaled_data, axis=-1), verbose=0)[0]
        cnn_pred = np.argmax(cnn_probs)
        
        rf_label = target_encoder.inverse_transform([rf_pred])[0]
        cnn_label = target_encoder.inverse_transform([cnn_pred])[0]
        
        latency = (time.time() - start_time) * 1000
        
    st.subheader("🎯 Real-Time Diagnostics Report")
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("### Random Forest Engine")
        if rf_label == "normal.":
            st.markdown(f"""<div class='metric-card-safe'>🟢 Network Status Verified:<br><span class='badge-text'>{rf_label.upper()}</span></div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class='metric-card-danger'>🚨 Threat Profile Isolated:<br><span class='badge-text'>{rf_label.upper()}</span></div>""", unsafe_allow_html=True)
            
    with col_right:
        st.markdown("### Deep Learning 1D-CNN Engine")
        if cnn_label == "normal.":
            st.markdown(f"""<div class='metric-card-safe'>🟢 Packet Evaluation Normal:<br><span class='badge-text'>{cnn_label.upper()} ({cnn_probs[cnn_pred]*100:.2f}%)</span></div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class='metric-card-danger'>🚨 Anomaly Signature Flags:<br><span class='badge-text'>{cnn_label.upper()} ({cnn_probs[cnn_pred]*100:.2f}%)</span></div>""", unsafe_allow_html=True)
            
    # --- PROBABILITY DISTRIBUTION PLOT ---
    st.write("---")
    st.markdown("### 📊 Threat Classification Confidence Vector Distribution")
    st.markdown("This bar chart displays the neural network's internal prediction confidence metrics mapped across all eligible security classes:")
    
    prob_df = pd.DataFrame({
        'Network Connection Profile': class_labels,
        'Model Confidence Level': list(cnn_probs)
    }).set_index('Network Connection Profile')
    
    st.bar_chart(prob_df, y='Model Confidence Level')
    st.toast(f"Analysis processed successfully in {latency:.2f} ms!", icon="⚡")