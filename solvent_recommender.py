import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Draw
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

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
        cached_data = joblib.load(DATA_CACHE_PATH)
        # Handle both old (2 items) and new (3 items) cache formats
        if len(cached_data) == 2:
            return (*cached_data, {})  # Add empty solvent_names if missing
        return cached_data
    
    try:
        # Load all data files
        kddb = pd.read_excel("KDDB.xlsx", sheet_name=None)
        dbdq = pd.read_excel("DBDQ.xlsx", sheet_name=None)
        dbdt = pd.read_excel("DBDT.xlsx", sheet_name=None)
        
        # Extract all unique solvent systems and their compositions
        system_compositions = {}
        solvent_names = {}
        
        # Process DBDQ and DBDT
        for db in [dbdq, dbdt]:
            for sheet_name, sheet_data in db.items():
                if sheet_name == "None" or not isinstance(sheet_name, str):
                    continue
                    
                # Clean system name
                system_name = sheet_name.strip()
                if not system_name:
                    continue
                
                # Store all compositions for this system
                compositions = []
                for _, row in sheet_data.iterrows():
                    try:
                        comp = str(row.get('Composition', '')).strip()
                        if comp:
                            compositions.append(comp)
                            
                            # Extract solvent names
                            solvents = []
                            for i in range(1, 5):
                                solvent = row.get(f'Solvent {i}', '')
                                if pd.notna(solvent) and str(solvent).strip():
                                    solvents.append(str(solvent).strip())
                            if solvents:
                                solvent_names[(system_name, comp)] = solvents
                    except:
                        continue
                
                # Ensure each system has at least a "Default" composition
                system_compositions[system_name] = sorted(list(set(compositions))) if compositions else ["Default"]
        
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
                    system_name = str(row['System']).strip()
                    composition = str(row['Composition']).strip()
                    
                    if system_name in system_compositions:
                        # Use default composition if none specified
                        if not composition:
                            composition = system_compositions[system_name][0]
                        
                        if composition in system_compositions[system_name]:
                            # Find matching composition data
                            for db in [dbdq, dbdt]:
                                if system_name in db:
                                    system_data = db[system_name]
                                    match = system_data[system_data['Composition'].astype(str) == composition]
                                    
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
                                        break
                                    
                except Exception as e:
                    if debug_mode:
                        st.warning(f"Error processing row: {str(e)}")
                    continue
    
        df = pd.DataFrame(data) if data else pd.DataFrame()
        
        # Cache the processed data
        joblib.dump((df, system_compositions, solvent_names), DATA_CACHE_PATH)
        
        return df, system_compositions, solvent_names
        
    except Exception as e:
        st.error(f"Error loading files: {str(e)}")
        return pd.DataFrame(), {}, {}

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

def get_composition_data(system_name, composition):
    """Load composition data for a specific system."""
    try:
        dbdq = pd.read_excel("DBDQ.xlsx", sheet_name=None)
        dbdt = pd.read_excel("DBDT.xlsx", sheet_name=None)
        
        for db in [dbdq, dbdt]:
            if system_name in db:
                system_data = db[system_name]
                match = system_data[system_data['Composition'].astype(str) == str(composition)]
                if not match.empty:
                    return match.iloc[0]
                
        # If no exact match found, return first row for the system
        for db in [dbdq, dbdt]:
            if system_name in db:
                system_data = db[system_name]
                if not system_data.empty:
                    return system_data.iloc[0]
    except Exception as e:
        if debug_mode:
            st.warning(f"Error loading composition data: {str(e)}")
    return None

def main():
    # Load data
    with st.spinner("Loading and preprocessing data..."):
        training_data, system_compositions, solvent_names = load_and_preprocess_data(use_cached_data)
        
        # Ensure system_compositions is always a dictionary
        if not isinstance(system_compositions, dict):
            if isinstance(system_compositions, list):
                system_compositions = {sys: ["Default"] for sys in system_compositions} if system_compositions else {}
            else:
                system_compositions = {}
        
        if debug_mode:
            st.write(f"Loaded {len(training_data)} training samples")
            st.write(f"Found {len(system_compositions)} solvent systems")
            st.write("Sample systems:", list(system_compositions.keys())[:3] if system_compositions else "None")
        
        if training_data.empty:
            st.error("No valid training data found!")
            return
    
    # Check data quality
    comp_features = [col for col in training_data.columns 
                    if any(col.startswith(p) for p in ['%Vol', '%Mol', '%Mas'])]
    
    if not comp_features:
        st.error("No composition features found in data!")
        return
    
    # Train or load model
    if os.path.exists(MODEL_PATH) and st.sidebar.checkbox("Use cached model", value=True):
        try:
            model = joblib.load(MODEL_PATH)
            st.sidebar.success("Loaded pre-trained model")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return
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
    
    if not system_compositions:
        st.error("No solvent systems found in data files!")
        return
    
    # Filter out systems with no compositions
    valid_systems = {k: v for k, v in system_compositions.items() if v}
    if not valid_systems:
        st.error("No valid solvent systems found (all systems have no compositions)!")
        return
    
    # Select system
    selected_system = st.selectbox(
        "Select solvent system:",
        sorted(valid_systems.keys()),
        index=0
    )
    
    # Get available compositions for selected system
    compositions = valid_systems.get(selected_system, ["Default"])
    selected_composition = st.selectbox(
        "Select composition:",
        compositions,
        index=0
    )
    
    # Get composition data
    composition_data = get_composition_data(selected_system, selected_composition)
    if composition_data is None:
        st.warning(f"Could not load detailed composition data for {selected_system} - using default values")
        composition_data = {}
    
    # Display solvent composition
    st.subheader("Solvent Composition")
    
    # Get solvent names for this composition
    solvents = solvent_names.get((selected_system, selected_composition), [])
    
    if not solvents:
        # Try to extract from composition data
        solvents = []
        for i in range(1, 5):
            solvent = composition_data.get(f'Solvent {i}', '')
            if pd.notna(solvent) and str(solvent).strip():
                solvents.append(str(solvent).strip())
    
    if not solvents:
        st.warning("No solvent names found for this composition")
    else:
        for i, solvent in enumerate(solvents, 1):
            perc = composition_data.get(f'%Vol{i} - UP', 0)
            try:
                perc_value = float(perc)
                st.write(f"- {solvent}: {perc_value:.1f}%")
            except:
                st.write(f"- {solvent}: Percentage not available")
    
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
                    try:
                        input_features[f"{prefix}{i}"] = float(composition_data[col])
                    except:
                        input_features[f"{prefix}{i}"] = 0.0
                else:
                    input_features[f"{prefix}{i}"] = 0.0
        
        # Create DataFrame with correct column order
        try:
            input_df = pd.DataFrame(columns=model.feature_names_in_)
            for feat in model.feature_names_in_:
                input_df[feat] = [input_features.get(feat, 0)]
            
            # Debug
            if debug_mode:
                st.write("Input features:", input_features)
                st.write("Input DataFrame:", input_df)
            
            # Make prediction
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
            if debug_mode:
                st.write("Model features:", model.feature_names_in_)
                st.write("Input features:", input_features.keys())

if __name__ == "__main__":
    main()
