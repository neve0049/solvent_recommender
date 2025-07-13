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
st.set_page_config(page_title="🔬 Solvent System Recommender Pro", layout="wide")
st.title("🔬 Solvent System Recommender Pro")

# Constants
MODEL_PATH = "solvent_model.pkl"
DATA_CACHE_PATH = "processed_data_cache.pkl"

# Solvent database
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
    input_template = {
        'MolWeight': 0, 'LogP': 0, 'HBD': 0, 'HBA': 0,
        'TPSA': 0, 'RotatableBonds': 0, 'AromaticRings': 0,
        '%Vol1': 0, '%Vol2': 0, '%Vol3': 0, '%Vol4': 0,
        '%Mol1': 0, '%Mol2': 0, '%Mol3': 0, '%Mol4': 0,
        '%Mas1': 0, '%Mas2': 0, '%Mas3': 0, '%Mas4': 0
    }
    
    input_template.update(mol_features)
    
    perc_type = solvent_data['percentage_type']
    percentages = solvent_data['percentages']
    
    # Assign percentages to correct columns
    for i in range(len(percentages)):
        col_name = f"{perc_type}{i+1}"
        input_template[col_name] = percentages[i]
    
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
    
    # Percentage type selector
    percentage_type = st.radio(
        "Percentage type:",
        ['%Vol', '%Mol', '%Mas'],
        horizontal=True
    )
    
    # Solvent selection
    num_solvents = st.slider("Number of solvents", 1, 4, 1)
    
    solvents = []
    percentages = []
    remaining_perc = 100.0
    
    for i in range(num_solvents):
        col1, col2 = st.columns([3, 2])
        
        with col1:
            solvent = st.selectbox(
                f"Solvent {i+1}",
                list(SOLVENT_DB.keys()),
                key=f"solv_{i}"
            )
            solvents.append(solvent)
        
        with col2:
            if i == num_solvents - 1:
                # Last solvent gets remaining percentage
                percentage = st.number_input(
                    f"Percentage {solvent}",
                    min_value=0.0,
                    max_value=remaining_perc,
                    value=remaining_perc,
                    key=f"perc_{i}"
                )
            else:
                max_val = min(100.0, remaining_perc)
                percentage = st.number_input(
                    f"Percentage {solvent}",
                    min_value=0.0,
                    max_value=max_val,
                    value=(100.0/num_solvents) if i == 0 else 0.0,
                    key=f"perc_{i}"
                )
            
            percentages.append(percentage)
            remaining_perc -= percentage
    
    # Prepare solvent data
    solvent_data = {
        'names': solvents,
        'percentages': percentages,
        'percentage_type': percentage_type
    }
    
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
                f"{p}% {n}" for n, p in zip(solvents, percentages)
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
