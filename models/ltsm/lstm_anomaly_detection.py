import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

print("=" * 80)
print("ENHANCED LSTM AUTOENCODER ANOMALY DETECTION SYSTEM")
print("Intelligent Hard Water Management System v2.0")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    'seq_length': 30,
    'latent_dim': 8,
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.001,
    'threshold_percentile': 97,  # More sensitive than 95
    'min_anomaly_score': 1.0,
}

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[STEP 1] Loading datasets from Excel file...")

try:
    excel_file = 'LSTM_Training_Dataset_Final.xlsx'
    xl_file = pd.ExcelFile(excel_file)
    print(f"✓ Found Excel file with sheets: {xl_file.sheet_names}")
    
    if len(xl_file.sheet_names) >= 3:
        df1 = pd.read_excel(excel_file, sheet_name=0)
        df2 = pd.read_excel(excel_file, sheet_name=1)
        df3 = pd.read_excel(excel_file, sheet_name=2)
        data = pd.concat([df1, df2, df3], axis=1)
        data = data.loc[:, ~data.columns.duplicated()]
    else:
        data = pd.read_excel(excel_file, sheet_name=0)
    
    print(f"✓ Dataset shape: {data.shape}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

# ============================================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================================
print("\n[STEP 2] Feature engineering and preprocessing...")

feature_columns = [
    'Temperature (°C)', 'Humidity (%)', 'Pressure (hPa)', 
    'WindSpeed (km/h)', 'Target_Rainfall (mm)',
    'pH', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
    'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity',
    'Temperature_pred (°C)', 'Rainfall_pred (mm)', 
    'Hardness_prev_day', 'refinedhardnessforecast'
]

available_features = [f for f in feature_columns if f in data.columns]
feature_data = data[available_features].copy()

# Enhanced missing value handling
if feature_data.isnull().sum().sum() > 0:
    feature_data = feature_data.interpolate(method='linear').fillna(method='bfill')

# Add temporal features
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'])
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    data['Month'] = data['Date'].dt.month
    
    temporal_features = ['DayOfWeek', 'Month']
    for temp_feat in temporal_features:
        if temp_feat in data.columns:
            feature_data[temp_feat] = data[temp_feat]
            available_features.append(temp_feat)

print(f"✓ Total features: {len(available_features)}")
print(f"✓ Features: {', '.join(available_features[:5])}... (+{len(available_features)-5} more)")

# Normalize
scaler = StandardScaler()
data_scaled = scaler.fit_transform(feature_data)

# ============================================================================
# STEP 3: CREATE SEQUENCES
# ============================================================================
print("\n[STEP 3] Creating sequences...")

def create_sequences(data, seq_length=30, stride=1):
    """Create sequences with configurable stride"""
    sequences = []
    for i in range(0, len(data) - seq_length, stride):
        sequences.append(data[i:i+seq_length])
    return np.array(sequences)

X = create_sequences(data_scaled, seq_length=CONFIG['seq_length'], stride=1)
print(f"✓ Sequences shape: {X.shape}")

# ============================================================================
# STEP 4: SPLIT DATA
# ============================================================================
split_idx = int(0.8 * len(X))
X_train = X[:split_idx]
X_test = X[split_idx:]

print(f"\n[STEP 4] Data split:")
print(f"✓ Training: {X_train.shape[0]} samples")
print(f"✓ Testing: {X_test.shape[0]} samples")

# ============================================================================
# STEP 5: BUILD IMPROVED MODEL (STABLE VERSION)
# ============================================================================
print("\n[STEP 5] Building improved LSTM Autoencoder...")

def build_improved_autoencoder(seq_length, n_features, latent_dim=8):
    """
    Improved LSTM Autoencoder with Bidirectional layers
    More stable than attention-based approach
    """
    
    # Encoder
    encoder_input = layers.Input(shape=(seq_length, n_features), name='encoder_input')
    
    # Use Bidirectional LSTM for better temporal patterns
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, name='enc_lstm1'))(encoder_input)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True, name='enc_lstm2'))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(16, return_sequences=False, name='enc_lstm3'))(x)
    
    # Latent bottleneck
    latent = layers.Dense(latent_dim, activation='relu', name='latent')(x)
    
    # Decoder
    x = layers.RepeatVector(seq_length)(latent)
    x = layers.Bidirectional(layers.LSTM(16, return_sequences=True, name='dec_lstm1'))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True, name='dec_lstm2'))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, name='dec_lstm3'))(x)
    
    decoder_output = layers.TimeDistributed(layers.Dense(n_features), name='output')(x)
    
    model = Model(encoder_input, decoder_output, name='Improved_LSTM_Autoencoder')
    return model

model = build_improved_autoencoder(
    seq_length=CONFIG['seq_length'],
    n_features=X.shape[2],
    latent_dim=CONFIG['latent_dim']
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']),
    loss='mse',
    metrics=['mae']
)

print(f"✓ Model parameters: {model.count_params():,}")
model.summary()

# ============================================================================
# STEP 6: TRAIN WITH ENHANCED CALLBACKS
# ============================================================================
print("\n[STEP 6] Training model...")

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        'lstm_enhanced_best.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=0
    )
]

history = model.fit(
    X_train, X_train,
    epochs=CONFIG['epochs'],
    batch_size=CONFIG['batch_size'],
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

model.save('lstm_enhanced_final.keras')
print("✓ Model saved as 'lstm_enhanced_final.keras'")

# ============================================================================
# STEP 7: DYNAMIC THRESHOLD CALCULATION
# ============================================================================
print("\n[STEP 7] Calculating dynamic thresholds...")

X_pred = model.predict(X, verbose=0)
reconstruction_error = np.mean(np.square(X - X_pred), axis=(1, 2))

train_errors = reconstruction_error[:split_idx]
test_errors = reconstruction_error[split_idx:]

print(f"✓ Error statistics:")
print(f"  - Train: mean={train_errors.mean():.6f}, std={train_errors.std():.6f}")
print(f"  - Test:  mean={test_errors.mean():.6f}, std={test_errors.std():.6f}")

# Calculate multiple thresholds
thresholds = {
    'Mean+2σ': train_errors.mean() + 2 * train_errors.std(),
    'Mean+3σ': train_errors.mean() + 3 * train_errors.std(),
    '95th percentile': np.percentile(train_errors, 95),
    '97th percentile': np.percentile(train_errors, 97),
    '99th percentile': np.percentile(train_errors, 99),
    'IQR method': np.percentile(train_errors, 75) + 1.5 * (
        np.percentile(train_errors, 75) - np.percentile(train_errors, 25)
    )
}

print(f"\n✓ Threshold options:")
for name, value in thresholds.items():
    n_anomalies = (reconstruction_error > value).sum()
    pct = (n_anomalies / len(reconstruction_error)) * 100
    print(f"  - {name:20s}: {value:.6f} ({n_anomalies} anomalies, {pct:.2f}%)")

# Use 97th percentile (more sensitive than 95th)
THRESHOLD = thresholds[f'{CONFIG["threshold_percentile"]}th percentile']
print(f"\n✓ Selected threshold: {THRESHOLD:.6f}")

# ============================================================================
# STEP 8: FEATURE-LEVEL ANOMALY DETECTION
# ============================================================================
print("\n[STEP 8] Feature-level anomaly analysis...")

# Calculate per-feature reconstruction errors
feature_errors = np.mean(np.square(X - X_pred), axis=1)

# Detect anomalies
anomalies = reconstruction_error > THRESHOLD
anomaly_scores = reconstruction_error / THRESHOLD

# Enhanced severity classification (more granular)
def classify_severity(score):
    if score < 1.0:
        return 'Normal'
    elif score < 1.15:
        return 'Low'
    elif score < 1.3:
        return 'Medium'
    elif score < 1.5:
        return 'High'
    else:
        return 'Critical'

results = pd.DataFrame({
    'Index': range(len(reconstruction_error)),
    'Reconstruction_Error': reconstruction_error,
    'Anomaly_Score': anomaly_scores,
    'Is_Anomaly': anomalies,
    'Severity': [classify_severity(s) for s in anomaly_scores]
})

if 'Date' in data.columns:
    dates = data['Date'].iloc[CONFIG['seq_length']:CONFIG['seq_length'] + len(results)].values
    results['Date'] = dates

# Identify problematic features for each anomaly
print("✓ Identifying problematic features for anomalies...")
for idx in results[results['Is_Anomaly']].index:
    feature_err = feature_errors[idx]
    top_features_idx = np.argsort(feature_err)[-3:][::-1]
    top_features = [available_features[i] for i in top_features_idx]
    results.loc[idx, 'Top_Problematic_Features'] = ', '.join(top_features)

n_anomalies = anomalies.sum()
anomaly_rate = (n_anomalies / len(anomalies)) * 100

print(f"\n✓ Detection complete:")
print(f"  - Total samples: {len(anomalies)}")
print(f"  - Anomalies detected: {n_anomalies} ({anomaly_rate:.2f}%)")

# ============================================================================
# STEP 9: GENERATE CONTEXTUAL ALERTS
# ============================================================================
print("\n[STEP 9] Generating contextual alerts...")

def generate_alert(row):
    """Generate contextual alert with recommended actions"""
    alert = {
        'timestamp': row.get('Date', 'N/A'),
        'severity': row['Severity'],
        'score': f"{row['Anomaly_Score']:.2f}x",
        'features': row.get('Top_Problematic_Features', 'N/A'),
        'actions': []
    }
    
    # Context-aware recommendations based on problematic features
    features_str = str(row.get('Top_Problematic_Features', ''))
    
    if 'pH' in features_str:
        alert['actions'].append('Check pH levels and adjust chemical dosing system')
    if 'Hardness' in features_str:
        alert['actions'].append('Inspect water softener and regeneration cycle')
    if 'Turbidity' in features_str:
        alert['actions'].append('Check filtration system and perform backwash if needed')
    if 'Chloramines' in features_str or 'Chlorine' in features_str:
        alert['actions'].append('Verify chlorination system operation and dosage')
    if 'Conductivity' in features_str:
        alert['actions'].append('Check for dissolved solids and mineral content')
    if 'Temperature' in features_str:
        alert['actions'].append('Monitor temperature sensors and cooling systems')
    if 'Sulfate' in features_str:
        alert['actions'].append('Test for sulfate contamination sources')
    if 'Organic_carbon' in features_str:
        alert['actions'].append('Check for organic contamination and biofilm')
    
    # Severity-based priority
    if row['Severity'] == 'Critical':
        alert['actions'].insert(0, '🚨 IMMEDIATE ACTION REQUIRED - System shutdown may be necessary')
    elif row['Severity'] == 'High':
        alert['actions'].insert(0, '⚠️ HIGH PRIORITY - Inspect within 4 hours')
    elif row['Severity'] == 'Medium':
        alert['actions'].insert(0, '⚠️ MEDIUM PRIORITY - Schedule inspection within 24 hours')
    else:
        alert['actions'].insert(0, 'ℹ️ LOW PRIORITY - Monitor and log for trends')
    
    return alert

# Generate alerts for all anomalies
alerts = []
for idx, row in results[results['Is_Anomaly']].iterrows():
    alerts.append(generate_alert(row))

print(f"✓ Generated {len(alerts)} contextual alerts")

# ============================================================================
# STEP 10: ENHANCED VISUALIZATION
# ============================================================================
print("\n[STEP 10] Creating enhanced visualizations...")

fig = plt.figure(figsize=(20, 14))

# Plot 1: Training History
ax1 = plt.subplot(3, 3, 1)
plt.plot(history.history['loss'], label='Train Loss', linewidth=2, alpha=0.8)
plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2, alpha=0.8)
plt.xlabel('Epoch', fontsize=11)
plt.ylabel('Loss (MSE)', fontsize=11)
plt.title('Model Training History', fontsize=13, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(alpha=0.3)

# Plot 2: Error Distribution Comparison
ax2 = plt.subplot(3, 3, 2)
plt.hist(train_errors, bins=50, alpha=0.6, label='Train', density=True, color='blue', edgecolor='black')
plt.hist(test_errors, bins=50, alpha=0.6, label='Test', density=True, color='orange', edgecolor='black')
plt.axvline(THRESHOLD, color='red', linestyle='--', linewidth=2.5, label=f'Threshold ({THRESHOLD:.4f})')
plt.xlabel('Reconstruction Error', fontsize=11)
plt.ylabel('Density', fontsize=11)
plt.title('Error Distribution (Train vs Test)', fontsize=13, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(alpha=0.3)

# Plot 3: Anomaly Timeline with Severity Colors
ax3 = plt.subplot(3, 1, 2)
severity_colors = {
    'Normal': 'lightblue', 
    'Low': 'yellow', 
    'Medium': 'orange', 
    'High': 'red', 
    'Critical': 'darkred'
}

for severity in ['Normal', 'Low', 'Medium', 'High', 'Critical']:
    mask = results['Severity'] == severity
    if mask.any():
        plt.scatter(results[mask].index, results[mask]['Reconstruction_Error'],
                   c=severity_colors[severity], label=severity, alpha=0.7, s=40, edgecolors='black', linewidth=0.5)

plt.axhline(THRESHOLD, color='red', linestyle='--', linewidth=2.5, label=f'Threshold', zorder=10)
plt.xlabel('Sample Index (Time)', fontsize=11)
plt.ylabel('Reconstruction Error', fontsize=11)
plt.title('Anomaly Detection Timeline (Colored by Severity)', fontsize=13, fontweight='bold')
plt.legend(fontsize=10, loc='upper left')
plt.grid(alpha=0.3)

# Plot 4: Severity Distribution Bar Chart
ax4 = plt.subplot(3, 3, 7)
severity_counts = results['Severity'].value_counts()
bar_colors = [severity_colors[s] for s in severity_counts.index]
bars = plt.bar(severity_counts.index, severity_counts.values, 
               color=bar_colors, edgecolor='black', linewidth=1.5)
plt.xlabel('Severity Level', fontsize=11)
plt.ylabel('Count', fontsize=11)
plt.title('Anomaly Severity Distribution', fontsize=13, fontweight='bold')
plt.xticks(rotation=45)
plt.grid(alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 5: Top Features Contributing to Anomalies
ax5 = plt.subplot(3, 3, 8)
if anomalies.any():
    anomaly_features = feature_errors[anomalies]
    top_n = min(10, len(available_features))
    mean_errors = anomaly_features.mean(axis=0)
    top_features_idx = np.argsort(mean_errors)[-top_n:][::-1]
    
    feature_names = [available_features[i] for i in top_features_idx]
    feature_values = mean_errors[top_features_idx]
    
    plt.barh(feature_names, feature_values, color='coral', edgecolor='black', linewidth=1)
    plt.xlabel('Mean Reconstruction Error', fontsize=11)
    plt.ylabel('Feature', fontsize=10)
    plt.title('Top Features Contributing to Anomalies', fontsize=13, fontweight='bold')
    plt.grid(alpha=0.3, axis='x')

# Plot 6: Cumulative Anomalies Over Time
ax6 = plt.subplot(3, 3, 9)
cumulative_anomalies = np.cumsum(anomalies)
plt.plot(cumulative_anomalies, linewidth=2.5, color='red', alpha=0.8)
plt.fill_between(range(len(cumulative_anomalies)), cumulative_anomalies, alpha=0.3, color='red')
plt.xlabel('Sample Index (Time)', fontsize=11)
plt.ylabel('Cumulative Anomaly Count', fontsize=11)
plt.title('Cumulative Anomalies Detected', fontsize=13, fontweight='bold')
plt.grid(alpha=0.3)

# Plot 7: MAE over epochs
ax7 = plt.subplot(3, 3, 3)
plt.plot(history.history['mae'], label='Train MAE', linewidth=2, alpha=0.8)
plt.plot(history.history['val_mae'], label='Val MAE', linewidth=2, alpha=0.8)
plt.xlabel('Epoch', fontsize=11)
plt.ylabel('MAE', fontsize=11)
plt.title('Mean Absolute Error History', fontsize=13, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('enhanced_anomaly_detection.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved as 'enhanced_anomaly_detection.png'")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n[STEP 11] Saving results...")

results.to_csv('enhanced_anomaly_results.csv', index=False)
print("✓ Results saved to 'enhanced_anomaly_results.csv'")

# Save detailed alerts
if alerts:
    alerts_df = pd.DataFrame(alerts)
    alerts_df.to_csv('contextual_alerts.csv', index=False)
    print("✓ Alerts saved to 'contextual_alerts.csv'")

# Save feature contribution analysis
if anomalies.any():
    anomaly_features = feature_errors[anomalies]
    mean_errors = anomaly_features.mean(axis=0)
    feature_contribution = pd.DataFrame({
        'Feature': available_features,
        'Mean_Error': mean_errors,
        'Contribution_Pct': (mean_errors / mean_errors.sum()) * 100
    })
    feature_contribution = feature_contribution.sort_values('Mean_Error', ascending=False)
    feature_contribution.to_csv('feature_anomaly_contribution.csv', index=False)
    print("✓ Feature analysis saved to 'feature_anomaly_contribution.csv'")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("ENHANCED ANOMALY DETECTION - FINAL SUMMARY")
print("=" * 80)
print(f"✓ Model: Improved Bidirectional LSTM Autoencoder")
print(f"✓ Total Parameters: {model.count_params():,}")
print(f"✓ Features Used: {len(available_features)}")
print(f"✓ Sequence Length: {CONFIG['seq_length']} days")
print(f"✓ Detection Threshold: {THRESHOLD:.6f} ({CONFIG['threshold_percentile']}th percentile)")
print(f"✓ Total Samples Analyzed: {len(anomalies)}")
print(f"✓ Anomalies Detected: {n_anomalies} ({anomaly_rate:.2f}%)")

print(f"\n✓ Severity Breakdown:")
for severity in ['Low', 'Medium', 'High', 'Critical']:
    count = (results['Severity'] == severity).sum()
    if count > 0:
        pct = (count / len(results)) * 100
        print(f"  - {severity:8s}: {count:3d} samples ({pct:.2f}%)")

# Show top 5 critical anomalies
if n_anomalies > 0:
    print(f"\n🚨 TOP 5 MOST CRITICAL ANOMALIES:")
    print("=" * 80)
    top_alerts = results[results['Is_Anomaly']].nlargest(5, 'Anomaly_Score')
    
    for i, (idx, row) in enumerate(top_alerts.iterrows(), 1):
        print(f"\n  #{i} - {row['Severity'].upper()} SEVERITY")
        if 'Date' in row and pd.notna(row['Date']):
            print(f"      Date: {row['Date']}")
        print(f"      Anomaly Score: {row['Anomaly_Score']:.2f}x threshold")
        print(f"      Reconstruction Error: {row['Reconstruction_Error']:.6f}")
        if 'Top_Problematic_Features' in row and pd.notna(row['Top_Problematic_Features']):
            print(f"      Problematic Features: {row['Top_Problematic_Features']}")
        
        # Show recommended actions
        alert = generate_alert(row)
        if alert['actions']:
            print(f"      Recommended Actions:")
            for action in alert['actions'][:3]:  # Show first 3 actions
                print(f"        • {action}")

print("\n" + "=" * 80)
print("📁 OUTPUT FILES GENERATED:")
print("=" * 80)
print("  1. lstm_enhanced_final.keras - Trained model")
print("  2. lstm_enhanced_best.h5 - Best checkpoint model")
print("  3. enhanced_anomaly_results.csv - Complete detection results")
print("  4. contextual_alerts.csv - Actionable alerts with recommendations")
print("  5. feature_anomaly_contribution.csv - Feature-level analysis")
print("  6. enhanced_anomaly_detection.png - Comprehensive visualizations")

print("\n✅ Enhanced Anomaly Detection System Complete!")
print("=" * 80)