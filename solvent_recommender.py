import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Draw
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
st.set_page_config(page_title="🔬 Advanced Solvent Recommender", layout="wide")
st.title("🔬 Advanced Solvent System Recommender")

# Constants
MODEL_PATH = "solvent_model.pkl"
DATA_CACHE_PATH = "processed_data_cache.pkl"

# Enhanced solvent database
SOLVENT_DB = {
    'Water': {'SMILES': 'O', 'Type': 'Polar'},
    'Methanol': {'SMILES': 'CO', 'Type': 'Polar protic'},
    'Ethanol': {'SMILES': 'CCO', 'Type': 'Polar protic'},
    'Acetonitrile': {'SMILES': 'CC#N', 'Type': 'Polar aprotic'},
    'Chloroform': {'SMILES': 'ClC(Cl)Cl', 'Type': 'Non-polar'},
    'Hexane': {'SMILES': 'CCCCCC', 'Type': 'Non-polar'},
    'Acetone': {'SMILES': 'CC(=O)C', 'Type': 'Polar aprotic'},
    'DMSO': {'SMILES': 'CS(=O)C', 'Type': 'Polar aprotic'}
}

# Debug mode
debug_mode = st.sidebar.checkbox("Debug Mode", False)

def load_model():
    """Load pre-trained model with error handling."""
    try:
        if os.path.exists(MODEL_PATH):
            return joblib.load(MODEL_PATH)
        else:
            st.error("Model file not found! Please train the model first.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def calculate_molecular_features(smiles):
    """Calculate molecular features from SMILES with validation."""
    if not smiles:
        return None
    
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        st.error("Invalid SMILES string")
        return None
    
    return {
        'MolWeight': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'HBD': Lipinski.NumHDonors(mol),
        'HBA': Lipinski.NumHAcceptors(mol),
        'TPSA': Descriptors.TPSA(mol),
        'RotatableBonds': Lipinski.NumRotatableBonds(mol),
        'AromaticRings': Lipinski.NumAromaticRings(mol)
    }

def create_input_vector(mol_features, solvent_data):
    """Create proper input vector for the model."""
    # Base template with zeros for all expected features
    input_template = {
        'MolWeight': 0,
        'LogP': 0,
        'HBD': 0,
        'HBA': 0,
        'TPSA': 0,
        'RotatableBonds': 0,
        'AromaticRings': 0,
        '%Vol1': 0, '%Vol2': 0, '%Vol3': 0, '%Vol4': 0,
        '%Mol1': 0, '%Mol2': 0, '%Mol3': 0, '%Mol4': 0,
        '%Mas1': 0, '%Mas2': 0, '%Mas3': 0, '%Mas4': 0
    }
    
    # Update with molecular features
    input_template.update(mol_features)
    
    # Update with solvent composition
    for i, (perc_type, value) in enumerate(solvent_data['percentages'].items(), 1):
        input_template[f"{perc_type}{i}"] = value
    
    return pd.DataFrame([input_template])

def main():
    # Load model
    model = load_model()
    if not model:
        return

    # Input section
    st.header("1. Molecule Input")
    smiles = st.text_input("Enter compound SMILES:", "CCO")
    
    if not smiles:
        st.warning("Please enter a SMILES string")
        return
    
    # Calculate molecular features
    mol_features = calculate_molecular_features(smiles)
    if not mol_features:
        return
    
    # Display molecule
    st.subheader("Molecular Structure")
    img = Draw.MolToImage(Chem.MolFromSmiles(smiles))
    st.image(img, caption=f"SMILES: {smiles}", width=300)

    # Solvent selection
    st.header("2. Solvent Composition")
    
    # Create 4 solvent slots
    cols = st.columns(4)
    solvent_data = {
        'names': [],
        'percentages': {},
        'percentage_type': '%Vol'  # Default to volume percentage
    }
    
    # Percentage type selector
    solvent_data['percentage_type'] = st.radio(
        "Percentage type:",
        ['%Vol', '%Mol', '%Mas'],
        horizontal=True
    )
    
    # Solvent selectors and percentages
    total_percentage = 0
    for i in range(4):
        with cols[i]:
            solvent_name = st.selectbox(
                f"Solvent {i+1}",
                ['None'] + list(SOLVENT_DB.keys()),
                key=f"solv_{i}"
            )
            
            if solvent_name != 'None':
                solvent_data['names'].append(solvent_name)
                
                if i == 0:
                    # First solvent defaults to 100%
                    percentage = 100.0
                else:
                    percentage = st.slider(
                        f"Percentage {solvent_name}",
                        0.0, 100.0, 0.0,
                        key=f"perc_{i}"
                    )
                
                solvent_data['percentages'][solvent_data['percentage_type']] = percentage
                total_percentage += percentage
    
    # Auto-balance percentages
    if len(solvent_data['names']) > 1 and total_percentage > 100:
        st.error("Total percentage cannot exceed 100%!")
        return
    
    # Prediction section
    st.header("3. Prediction")
    
    if st.button("Predict Log KD"):
        # Prepare input data
        input_df = create_input_vector(mol_features, solvent_data)
        
        if debug_mode:
            st.write("Model input features:")
            st.dataframe(input_df)
        
        # Make prediction
        try:
            prediction = model.predict(input_df)[0]
            
            # Display results
            st.success(f"Predicted Log KD: {prediction:.2f}")
            
            # Show solvent system
            system_desc = " + ".join([
                f"{p}% {n}" for n, p in zip(
                    solvent_data['names'],
                    solvent_data['percentages'].values()
                )
            ])
            st.write(f"Solvent System: {system_desc}")
            
            # Feature importance
            st.subheader("Feature Importance")
            importances = pd.DataFrame({
                'Feature': input_df.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=importances, ax=ax)
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    main()
