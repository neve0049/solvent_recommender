import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Draw
import numpy as np
from io import BytesIO
from PIL import Image

# Configuration
st.set_page_config(page_title="Solvent System Recommender", layout="wide")
st.title("🔬 Solvent System Recommender")

# Debug flag
debug_mode = st.checkbox("Enable debug mode")

# Debug function
def show_debug_data(data, name):
    if debug_mode:
        st.write(f"Debug - {name} columns: {data.columns.tolist()}")
        st.write(f"Debug - {name} first rows:", data.head())

# Load data functions
@st.cache_data
def load_data():
    try:
        # Load KDDB data
        kddb = pd.read_excel("KDDB.xlsx", sheet_name=None)
        if debug_mode:
            show_debug_data(pd.concat(kddb.values()), "KDDB")
        
        # Load solvent system data
        dbdq = pd.read_excel("DBDQ.xlsx", sheet_name=None)
        if debug_mode:
            show_debug_data(pd.concat(dbdq.values()), "DBDQ")
        
        dbdt = pd.read_excel("DBDT.xlsx", sheet_name=None)
        if debug_mode:
            show_debug_data(pd.concat(dbdt.values()), "DBDT")
        
        return kddb, dbdq, dbdt
    except Exception as e:
        st.error(f"Error loading files: {str(e)}")
        return None, None, None

def extract_features(smiles):
    """Calculate molecular features from SMILES"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        return {
            'MolWeight': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'HBD': Lipinski.NumHDonors(mol),
            'HBA': Lipinski.NumHAcceptors(mol),
            'TPSA': Descriptors.TPSA(mol),
            'RotatableBonds': Lipinski.NumRotatableBonds(mol),
            'AromaticRings': Lipinski.NumAromaticRings(mol),
            'HeavyAtoms': Lipinski.HeavyAtomCount(mol)
        }
    except:
        return None

def prepare_training_data(kddb, dbdq, dbdt):
    """Prepare training data from all Excel files"""
    data = []
    
    for sheet_name, sheet_data in kddb.items():
        for _, row in sheet_data.iterrows():
            try:
                # Skip if essential data is missing
                if pd.isna(row['SMILES']) or pd.isna(row['Log KD']) or pd.isna(row['System']):
                    continue
                    
                # Get molecular features
                features = extract_features(row['SMILES'])
                if features is None:
                    continue
                    
                # Get solvent system data
                system_name = row['System']
                composition = row['Composition']
                
                # Handle different composition types
                if isinstance(composition, str):
                    composition = composition.strip()
                else:
                    composition = str(int(composition)) if not pd.isna(composition) else ""
                
                solvent_data = None
                
                # Search in DBDQ and DBDT
                for db in [dbdq, dbdt]:
                    if system_name in db:
                        solvent_sheet = db[system_name]
                        solvent_row = solvent_sheet[solvent_sheet['Composition'].astype(str) == str(composition)]
                        if not solvent_row.empty:
                            solvent_data = solvent_row.iloc[0].to_dict()
                            break
                
                if not solvent_data:
                    continue
                    
                # Prepare record
                record = {
                    **features,
                    'Log_KD': float(row['Log KD']),
                    'System': system_name,
                    'Composition_Number': composition
                }
                
                # Add solvent composition features
                for col, val in solvent_data.items():
                    if any(col.startswith(prefix) for prefix in ['%Vol', '%Mol', '%Mas']):
                        try:
                            record[col] = float(val) if not pd.isna(val) else 0.0
                        except:
                            record[col] = 0.0
                
                data.append(record)
            except Exception as e:
                if debug_mode:
                    st.warning(f"Skipping row due to error: {str(e)}")
                continue
    
    return pd.DataFrame(data)

def train_model(df):
    """Train a RandomForest model on the prepared data"""
    if df.empty:
        st.error("Training data is empty!")
        return None, None
        
    # Select features and target
    feature_cols = [
        'MolWeight', 'LogP', 'HBD', 'HBA', 'TPSA', 
        'RotatableBonds', 'AromaticRings', 'HeavyAtoms'
    ]
    
    # Add solvent composition features
    solvent_features = [col for col in df.columns 
                       if any(col.startswith(prefix) for prefix in ['%Vol', '%Mol', '%Mas'])]
    feature_cols.extend(solvent_features)
    
    # Ensure we only use columns that exist
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols]
    y = df['Log_KD']
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, feature_cols

def predict_for_systems(model, features, feature_cols, dbdq, dbdt):
    """Predict log KD for all solvent systems"""
    results = []
    
    for system_name, system_data in {**dbdq, **dbdt}.items():
        for _, solvent_row in system_data.iterrows():
            try:
                # Prepare input features
                input_features = features.copy()
                input_features['System'] = system_name
                input_features['Composition_Number'] = str(solvent_row['Composition'])
                
                # Add solvent composition features
                for col in solvent_row.index:
                    if any(col.startswith(prefix) for prefix in ['%Vol', '%Mol', '%Mas']):
                        try:
                            input_features[col] = float(solvent_row[col]) if not pd.isna(solvent_row[col]) else 0.0
                        except:
                            input_features[col] = 0.0
                
                # Create DataFrame row
                input_df = pd.DataFrame([input_features])
                
                # Ensure all feature columns are present
                for col in feature_cols:
                    if col not in input_df:
                        input_df[col] = 0.0
                
                # Predict
                log_kd = model.predict(input_df[feature_cols])[0]
                
                if debug_mode:
                    st.write(f"System: {system_name}, Composition: {solvent_row['Composition']}, Predicted Log KD: {log_kd:.2f}")
                
                if -1 <= log_kd <= 1:
                    # Get composition details
                    composition = []
                    for i in range(1, 5):
                        vol_col = f'%Vol{i} - UP'
                        if vol_col in solvent_row and not pd.isna(solvent_row[vol_col]):
                            composition.append(f"{solvent_row[vol_col]:.3f}")
                    
                    solvent_names = []
                    for i in range(1, 5):
                        name_col = f'Solvent {i}'
                        if name_col in solvent_row and not pd.isna(solvent_row[name_col]):
                            solvent_names.append(str(solvent_row[name_col]))
                    
                    results.append({
                        "System": str(system_name),
                        "Composition": " / ".join(composition),
                        "Solvents": " / ".join(solvent_names),
                        "Composition Number": str(solvent_row['Composition']),
                        "Predicted Log KD": f"{log_kd:.2f}"
                    })
            except Exception as e:
                if debug_mode:
                    st.warning(f"Skipping prediction for {system_name}-{solvent_row['Composition']}: {str(e)}")
                continue
    
    return results

# Main app
def main():
    st.markdown("""
    Entrez une chaîne SMILES pour obtenir des recommandations de systèmes de solvants avec un log KD prédit entre -1 et 1.
    """)
    
    # Load data
    with st.spinner("Chargement des données..."):
        kddb, dbdq, dbdt = load_data()
    
    if kddb is None or dbdq is None or dbdt is None:
        st.error("Échec du chargement des fichiers de données. Veuillez vérifier que les fichiers existent et sont des fichiers Excel valides.")
        return
    
    # Prepare training data
    with st.spinner("Préparation des données d'entraînement..."):
        training_data = prepare_training_data(kddb, dbdq, dbdt)
        if debug_mode:
            show_debug_data(training_data, "Training Data")
    
    if training_data.empty:
        st.error("Aucune donnée d'entraînement valide n'a pu être préparée. Problèmes possibles :")
        st.write("- Les chaînes SMILES dans KDDB peuvent être invalides")
        st.write("- Les combinaisons Système/Composition peuvent ne pas correspondre entre les fichiers")
        st.write("- Valeurs de Log KD manquantes ou invalides")
        return
    
    # Train model
    with st.spinner("Entraînement du modèle..."):
        model, feature_cols = train_model(training_data)
    
    if model is None:
        st.error("Échec de l'entraînement du modèle.")
        return
    
    # Get user input
    st.subheader("Entrée de la molécule")
    smiles = st.text_input("Entrez une chaîne SMILES", value="C1=CC(=CC=C1/C=C/C2=CC(=CC(=C2)O)O)O")
    
    if st.button("Trouver des systèmes de solvants appropriés"):
        if not smiles:
            st.warning("Veuillez entrer une chaîne SMILES")
            return
            
        # Calculate molecular features
        with st.spinner("Calcul des caractéristiques moléculaires..."):
            features = extract_features(smiles)
        
        if features is None:
            st.error("Chaîne SMILES invalide. Veuillez entrer une chaîne SMILES valide.")
            return

        mol = Chem.MolFromSmiles(smiles)
        if mol:
            st.subheader("Structure moléculaire")
            img = Draw.MolToImage(mol, size=(400, 400))
            buf = BytesIO()
            img.save(buf, format="PNG")
            st.image(buf.getvalue(), width=200)

        # Faire les prédictions
        with st.spinner("Prédiction des systèmes de solvants..."):
            results = predict_for_systems(model, features, feature_cols, dbdq, dbdt)

        if results:
            st.subheader("🔍 Systèmes de solvants prédits")
            df_results = pd.DataFrame(results)
            
            # Convert all columns to string to avoid Arrow issues
            for col in df_results.columns:
                df_results[col] = df_results[col].astype(str)
            
            # Sort by proximity to 0 (best log KD)
            df_results['Abs Log KD'] = df_results['Predicted Log KD'].astype(float).abs()
            df_results = df_results.sort_values('Abs Log KD').drop('Abs Log KD', axis=1)
            
            st.dataframe(df_results, use_container_width=True)
            
            if debug_mode:
                st.write("Debug - Top predictions details:")
                st.write(df_results.head())
        else:
            st.warning("Aucun système de solvants prédit avec un log KD entre -1 et 1.")

if __name__ == "__main__":
    main()
