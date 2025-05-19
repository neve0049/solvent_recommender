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
    
    if debug_mode:
        st.write("Debug - Features being used:", feature_cols)
        st.write("Debug - Sample X values:", X.head())
        st.write("Debug - Sample y values:", y.head())
    
    # Train model with more trees and depth
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        random_state=42
    )
    model.fit(X, y)
    
    if debug_mode:
        st.write("Debug - Model feature importances:")
        st.write(dict(zip(feature_cols, model.feature_importances_)))
    
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
                
                # Reorder columns to match training data
                input_df = input_df[feature_cols]
                
                # Predict
                log_kd = model.predict(input_df)[0]
                
                if debug_mode:
                    st.write(f"System: {system_name}, Composition: {solvent_row['Composition']}, Predicted Log KD: {log_kd:.2f}")
                    st.write("Input features:", input_df.iloc[0].to_dict())
                
                if -1 <= log_kd <= 1:
                    # Get composition details
                    composition = []
                    solvent_names = []
                    for i in range(1, 5):
                        vol_col = f'%Vol{i} - UP'
                        name_col = f'Solvent {i}'
                        
                        if vol_col in solvent_row and not pd.isna(solvent_row[vol_col]):
                            composition.append(f"{solvent_row[vol_col]:.1f}%")
                        if name_col in solvent_row and not pd.isna(solvent_row[name_col]):
                            solvent_names.append(str(solvent_row[name_col]))
                    
                    results.append({
                        "System": str(system_name),
                        "Composition": " / ".join(composition),
                        "Solvents": " / ".join(solvent_names),
                        "Composition ID": str(solvent_row['Composition']),
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
            if not training_data.empty:
                st.write("Debug - Log KD value distribution:")
                st.write(training_data['Log_KD'].describe())
    
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
            
            # Convert KD to numeric for sorting
            df_results['KD_value'] = df_results['Predicted Log KD'].astype(float)
            
            # Sort by absolute value closest to 0 (best KD)
            df_results['Abs_KD'] = df_results['KD_value'].abs()
            df_results = df_results.sort_values('Abs_KD')
            
            # Format final display
            display_cols = ["System", "Solvents", "Composition", "Predicted Log KD"]
            st.dataframe(
                df_results[display_cols],
                column_config={
                    "Predicted Log KD": st.column_config.NumberColumn(
                        format="%.2f",
                        help="Valeur prédite du log KD (idéalement entre -1 et 1)"
                    )
                },
                use_container_width=True
            )
            
            if debug_mode:
                st.write("Debug - Full prediction results:", df_results)
        else:
            st.warning("Aucun système de solvants prédit avec un log KD entre -1 et 1.")
            if debug_mode:
                st.write("Debug - Features used for prediction:", features)
                st.write("Debug - Feature columns expected by model:", feature_cols)

if __name__ == "__main__":
    main()
