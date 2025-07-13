import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
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
st.set_page_config(page_title="Solvent System Recommender", layout="wide")
st.title("🔬 Advanced Solvent System Recommender")

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
        # Load all data files
        kddb = pd.read_excel("KDDB.xlsx", sheet_name=None)
        dbdq = pd.read_excel("DBDQ.xlsx", sheet_name=None)
        dbdt = pd.read_excel("DBDT.xlsx", sheet_name=None)
        
        # Extract all unique solvent systems
        all_systems = set()
        solvent_compositions = {}
        
        # Process DBDQ and DBDT to get solvent combinations
        for db in [dbdq, dbdt]:
            for sheet_name, sheet_data in db.items():
                system_name = sheet_name
                all_systems.add(system_name)
                
                # Store composition data for each system
                solvent_compositions[system_name] = sheet_data
        
        # Process KDDB to get molecular data
        data = []
        for sheet_name, sheet_data in kddb.items():
            for _, row in sheet_data.iterrows():
                try:
                    if pd.isna(row['SMILES']) or pd.isna(row['Log KD']) or pd.isna(row['System']):
                        continue
                        
                    mol = Chem.MolFromSmiles(row['SMILES'])
                    if mol is None:
                        continue
                        
                    # Calculate molecular features
                    features = {
                        'MolWeight': Descriptors.MolWt(mol),
                        'LogP': Descriptors.MolLogP(mol),
                        'HBD': Lipinski.NumHDonors(mol),
                        'HBA': Lipinski.NumHAcceptors(mol),
                        'TPSA': Descriptors.TPSA(mol),
                        'RotatableBonds': Lipinski.NumRotatableBonds(mol),
                        'AromaticRings': Lipinski.NumAromaticRings(mol),
                        'HeavyAtoms': Lipinski.HeavyAtomCount(mol),
                        'RingCount': Lipinski.RingCount(mol),
                        'FractionCSP3': Descriptors.FractionCSP3(mol),
                        'Log_KD': float(row['Log KD'])
                    }
                    
                    # Get solvent composition
                    system_name = row['System']
                    composition = str(row['Composition']).strip()
                    
                    if system_name in solvent_compositions:
                        system_data = solvent_compositions[system_name]
                        match = system_data[system_data['Composition'] == composition]
                        
                        if not match.empty:
                            solvent_data = match.iloc[0].to_dict()
                            
                            # Add composition features
                            for i in range(1, 5):
                                for prefix in ['%Vol', '%Mol', '%Mas']:
                                    col = f"{prefix}{i} - UP"
                                    if col in solvent_data:
                                        try:
                                            features[f"{prefix}{i}"] = float(solvent_data[col])
                                        except:
                                            features[f"{prefix}{i}"] = 0.0
                            
                            data.append(features)
                            
                except Exception as e:
                    if debug_mode:
                        st.warning(f"Error processing row: {str(e)}")
                    continue
    
        df = pd.DataFrame(data)
        
        # Cache the processed data
        joblib.dump((df, list(all_systems)), DATA_CACHE_PATH)
        
        return df, list(all_systems)
        
    except Exception as e:
        st.error(f"Error loading files: {str(e)}")
        return None, None

def train_model(data):
    """Train and evaluate the Random Forest model."""
    if data is None or len(data) == 0:
        return None, 0, 0
    
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
    # Load data
    with st.spinner("Loading and preprocessing data..."):
        training_data, all_systems = load_and_preprocess_data(use_cached_data)
        
        if training_data is None or all_systems is None:
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
            if model is not None:
                joblib.dump(model, MODEL_PATH)
                st.sidebar.success(f"Model trained (R²: {r2:.3f}, MSE: {mse:.3f})")
            else:
                st.error("Failed to train model!")
                return
    
    # Main interface
    st.header("1. Molecule Input")
    smiles = st.text_input("Enter SMILES string:", "CCO")
    
    if not smiles:
        st.warning("Please enter a SMILES string")
        return
    
    # Calculate molecular features
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        st.error("Invalid SMILES string")
        return
    
    mol_features = {
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
    
    # Display molecule
    st.subheader("Molecular Structure")
    visualize_molecule(smiles)
    
    # Solvent selection
    st.header("2. Solvent System Selection")
    
    # Select system
    selected_system = st.selectbox(
        "Select solvent system:",
        sorted(all_systems),
        index=0
    )
    
    # Load composition data for selected system
    system_data = None
    try:
        dbdq = pd.read_excel("DBDQ.xlsx", sheet_name=None)
        dbdt = pd.read_excel("DBDT.xlsx", sheet_name=None)
        
        if selected_system in dbdq:
            system_data = dbdq[selected_system]
        elif selected_system in dbdt:
            system_data = dbdt[selected_system]
    except:
        pass
    
    if system_data is None:
        st.error(f"Could not load data for system: {selected_system}")
        return
    
    # Get available compositions
    available_compositions = system_data['Composition'].unique()
    selected_composition = st.selectbox(
        "Select composition:",
        available_compositions,
        index=0
    )
    
    # Get the selected composition data
    composition_data = system_data[system_data['Composition'] == selected_composition].iloc[0]
    
    # Display solvent percentages
    st.subheader("Solvent Composition")
    
    # Extract solvent names
    solvents = []
    for i in range(1, 5):
        col = f"Solvent {i}"
        if col in composition_data and not pd.isna(composition_data[col]):
            solvents.append(composition_data[col])
    
    # Display percentages
    for i, solvent in enumerate(solvents, 1):
        perc = composition_data.get(f"%Vol{i} - UP", 0)
        st.write(f"- {solvent}: {perc:.1f}%")
    
    # Prediction
    st.header("3. Prediction")
    
    if st.button("Predict Log KD"):
        # Prepare input vector
        input_features = mol_features.copy()
        
        # Add composition features
        for i in range(1, 5):
            for prefix in ['%Vol', '%Mol', '%Mas']:
                col = f"{prefix}{i} - UP"
                if col in composition_data:
                    input_features[f"{prefix}{i}"] = float(composition_data[col])
                else:
                    input_features[f"{prefix}{i}"] = 0.0
        
        # Create DataFrame with correct column order
        input_df = pd.DataFrame(columns=model.feature_names_in_)
        for feat in model.feature_names_in_:
            input_df[feat] = [input_features.get(feat, 0)]
        
        # Make prediction
        try:
            prediction = model.predict(input_df)[0]
            st.success(f"Predicted Log KD: {prediction:.2f}")
            
            # Show feature importance
            st.subheader("Feature Importance")
            importances = pd.DataFrame({
                'Feature': model.feature_names_in_,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=importances, ax=ax)
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    main()
