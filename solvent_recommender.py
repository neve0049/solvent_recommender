import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Draw
import numpy as np
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Configuration
st.set_page_config(page_title="🔬 Solvent System Recommender", layout="wide", page_icon="🔬")
st.title("🔬 Solvent System Recommender")

# Constants
MODEL_PATH = "solvent_model.pkl"
DATA_CACHE_PATH = "processed_data_cache.pkl"

# Debug flag
debug_mode = st.sidebar.checkbox("Enable debug mode", value=False)
use_cached_data = st.sidebar.checkbox("Use cached data", value=True)

@st.cache_resource(show_spinner="Loading and preprocessing data...")
def load_and_preprocess_data(use_cache=True):
    """Load and preprocess data with caching option."""
    if use_cache and os.path.exists(DATA_CACHE_PATH):
        return joblib.load(DATA_CACHE_PATH)
    
    try:
        # Load data with validation
        kddb = pd.read_excel("KDDB.xlsx", sheet_name=None)
        dbdq = pd.read_excel("DBDQ.xlsx", sheet_name=None)
        dbdt = pd.read_excel("DBDT.xlsx", sheet_name=None)
        
        # Data validation
        validate_data_systems(kddb, dbdq, dbdt)
        
        # Prepare training data
        training_data, composition_stats = prepare_training_data(kddb, dbdq, dbdt)
        
        # Cache the processed data
        joblib.dump((training_data, composition_stats), DATA_CACHE_PATH)
        
        return training_data, composition_stats
        
    except Exception as e:
        st.error(f"Error loading files: {str(e)}")
        return None, None

def validate_data_systems(kddb, dbdq, dbdt):
    """Validate consistency between datasets."""
    if not debug_mode:
        return
    
    kddb_systems = set()
    for sheet in kddb.values():
        kddb_systems.update(sheet['System'].unique())
    
    db_systems = set()
    for db in [dbdq, dbdt]:
        for sheet in db.values():
            db_systems.update(sheet['System'].unique())
    
    with st.expander("Data Validation Report"):
        st.write(f"Total systems in KDDB: {len(kddb_systems)}")
        st.write(f"Total systems in DBDQ/DBDT: {len(db_systems)}")
        
        only_in_kddb = kddb_systems - db_systems
        only_in_db = db_systems - kddb_systems
        
        if only_in_kddb:
            st.warning(f"Systems in KDDB but not in DBDQ/DBDT: {only_in_kddb}")
        if only_in_db:
            st.warning(f"Systems in DBDQ/DBDT but not in KDDB: {only_in_db}")
        
        if not only_in_kddb and not only_in_db:
            st.success("All systems match between datasets!")

def prepare_training_data(kddb, dbdq, dbdt):
    """Prepare the training dataset with features and targets."""
    data = []
    composition_stats = []
    failed_molecules = 0
    
    for sheet_name, sheet_data in kddb.items():
        for _, row in sheet_data.iterrows():
            try:
                # Skip if essential data is missing
                if pd.isna(row['SMILES']) or pd.isna(row['Log KD']) or pd.isna(row['System']):
                    continue
                    
                # Get molecular features
                mol = Chem.MolFromSmiles(row['SMILES'])
                if mol is None:
                    failed_molecules += 1
                    continue
                    
                features = calculate_molecular_features(mol)
                features['Log_KD'] = float(row['Log KD'])
                
                # Get solvent composition
                system_name = row['System']
                composition = str(row['Composition']).strip()
                solvent_data = find_solvent_data(system_name, composition, dbdq, dbdt)
                
                if not solvent_data:
                    if debug_mode:
                        st.warning(f"No match for {system_name}-{composition}")
                    continue
                
                # Add composition features
                comp_features = extract_composition_features(solvent_data)
                record = {**features, **comp_features}
                data.append(record)
                
            except Exception as e:
                if debug_mode:
                    st.warning(f"Error processing row: {str(e)}")
                continue
    
    df = pd.DataFrame(data)
    
    if debug_mode:
        debug_data_analysis(df, composition_stats, failed_molecules)
    
    return df, composition_stats

def calculate_molecular_features(mol):
    """Calculate all molecular features for a given molecule."""
    return {
        'MolWeight': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'HBD': Lipinski.NumHDonors(mol),
        'HBA': Lipinski.NumHAcceptors(mol),
        'TPSA': Descriptors.TPSA(mol),
        'RotatableBonds': Lipinski.NumRotatableBonds(mol),
        'AromaticRings': Lipinski.NumAromaticRings(mol),
        'HeavyAtoms': Lipinski.HeavyAtomCount(mol),
        'RingCount': Lipinski.RingCount(mol),
        'FractionCSP3': Descriptors.FractionCSP3(mol)
    }

def find_solvent_data(system_name, composition, dbdq, dbdt):
    """Find solvent composition data in DBDQ or DBDT."""
    for db in [dbdq, dbdt]:
        if system_name in db:
            system_df = db[system_name]
            match = system_df[system_df['Composition'].astype(str) == composition]
            if not match.empty:
                return match.iloc[0].to_dict()
    return None

def extract_composition_features(solvent_data):
    """Extract composition features from solvent data."""
    comp_features = {}
    for i in range(1, 5):  # For up to 4 solvents
        for prefix in ['%Vol', '%Mol', '%Mas']:
            col = f"{prefix}{i} - UP"
            if col in solvent_data:
                try:
                    value = float(solvent_data[col])
                    comp_features[f"{prefix}{i}"] = value
                except (ValueError, TypeError):
                    comp_features[f"{prefix}{i}"] = 0.0
    return comp_features

def debug_data_analysis(df, composition_stats, failed_molecules):
    """Show debug information about the data."""
    with st.expander("Debug Data Analysis"):
        st.write(f"Total samples: {len(df)}")
        st.write(f"Failed to process {failed_molecules} molecules")
        
        st.subheader("Log KD Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['Log_KD'], kde=True, ax=ax)
        st.pyplot(fig)
        
        st.subheader("Molecular Features Correlation")
        mol_features = ['MolWeight', 'LogP', 'HBD', 'HBA', 'TPSA']
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df[mol_features + ['Log_KD']].corr(), annot=True, ax=ax)
        st.pyplot(fig)
        
        st.subheader("Missing Values")
        st.write(df.isnull().sum())

def train_model(data):
    """Train and evaluate the Random Forest model."""
    # Separate features and target
    X = data.drop(columns=['Log_KD'])
    y = data['Log_KD']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mse, r2

def visualize_molecule(smiles):
    """Generate and display molecule image."""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol, size=(400, 300))
        st.image(img, caption=f"Molecule: {smiles}")
    else:
        st.warning("Invalid SMILES string - cannot display molecule")

def main():
    st.sidebar.header("Settings")
    
    # Load data
    training_data, _ = load_and_preprocess_data(use_cached_data)
    if training_data is None:
        st.error("Failed to load data. Please check the input files.")
        return
    
    # Check data quality
    comp_features = [col for col in training_data.columns 
                    if any(col.startswith(p) for p in ['%Vol', '%Mol', '%Mas'])]
    
    if not comp_features:
        st.error("No composition features found in data!")
        return
    
    # Train or load model
    if os.path.exists(MODEL_PATH) and st.sidebar.checkbox("Use cached model", value=True):
        model = joblib.load(MODEL_PATH)
        st.sidebar.success("Loaded pre-trained model")
    else:
        with st.spinner("Training model..."):
            model, mse, r2 = train_model(training_data)
            joblib.dump(model, MODEL_PATH)
            st.sidebar.success(f"Model trained (R²: {r2:.3f}, MSE: {mse:.3f})")
    
    # Main interface
    st.header("Molecule Input")
    smiles = st.text_input("Enter SMILES string:", "CCO")
    
    if smiles:
        try:
            # Display molecule
            visualize_molecule(smiles)
            
            # Calculate features
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                features = calculate_molecular_features(mol)
                
                # Create input for prediction
                input_df = pd.DataFrame([features])
                
                # Add composition features with default values
                for col in comp_features:
                    input_df[col] = 0.0  # Will be replaced by user input
                
                # Display editable composition
                st.header("Solvent Composition")
                cols = st.columns(3)
                
                composition = {}
                for i, prefix in enumerate(['%Vol', '%Mol', '%Mas']):
                    with cols[i]:
                        st.subheader(prefix)
                        for j in range(1, 5):
                            key = f"{prefix}{j}"
                            if key in comp_features:
                                composition[key] = st.slider(
                                    f"Solvent {j}", 
                                    min_value=0.0, 
                                    max_value=100.0,
                                    value=25.0 if j == 1 else 0.0,
                                    key=key
                                )
                
                # Update input with user composition
                for key, value in composition.items():
                    input_df[key] = value
                
                # Make prediction
                if st.button("Predict Log KD"):
                    prediction = model.predict(input_df)[0]
                    st.success(f"Predicted Log KD: {prediction:.2f}")
                    
                    # Show feature importance
                    st.header("Feature Importance")
                    importances = pd.DataFrame({
                        'Feature': input_df.columns,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig, ax = plt.subplots()
                    sns.barplot(x='Importance', y='Feature', data=importances.head(10), ax=ax)
                    st.pyplot(fig)
                    
            else:
                st.error("Invalid SMILES string - cannot process molecule")
                
        except Exception as e:
            st.error(f"Error processing molecule: {str(e)}")

if __name__ == "__main__":
    main()
