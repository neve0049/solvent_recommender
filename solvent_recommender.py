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

def show_debug_data(data, name):
    if debug_mode:
        st.write(f"Debug - {name} columns: {data.columns.tolist()}")
        st.write(f"Debug - {name} first rows:", data.head())

@st.cache_data
def load_data():
    try:
        kddb = pd.read_excel("KDDB.xlsx", sheet_name=None)
        if debug_mode:
            show_debug_data(pd.concat(kddb.values()), "KDDB")
        
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
    data = []
    
    for sheet_name, sheet_data in kddb.items():
        for _, row in sheet_data.iterrows():
            try:
                if pd.isna(row['SMILES']) or pd.isna(row['Log KD']) or pd.isna(row['System']):
                    continue
                    
                features = extract_features(row['SMILES'])
                if features is None:
                    continue
                
                system_name = row['System']
                composition = row['Composition']
                
                if isinstance(composition, str):
                    composition = composition.strip()
                else:
                    composition = str(int(composition)) if not pd.isna(composition) else ""
                
                solvent_data = None
                
                for db in [dbdq, dbdt]:
                    if system_name in db:
                        solvent_sheet = db[system_name]
                        solvent_row = solvent_sheet[solvent_sheet['Composition'].astype(str) == str(composition)]
                        if not solvent_row.empty:
                            solvent_data = solvent_row.iloc[0].to_dict()
                            break
                
                if not solvent_data:
                    continue
                    
                record = {
                    **features,
                    'Log_KD': float(row['Log KD']),
                    'System': system_name,
                    'Composition': composition
                }
                
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
    if df.empty:
        st.error("Training data is empty!")
        return None, None
        
    feature_cols = [
        'MolWeight', 'LogP', 'HBD', 'HBA', 'TPSA', 
        'RotatableBonds', 'AromaticRings', 'HeavyAtoms'
    ]
    
    solvent_features = [col for col in df.columns 
                       if any(col.startswith(prefix) for prefix in ['%Vol', '%Mol', '%Mas'])]
    feature_cols.extend(solvent_features)
    
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols]
    y = df['Log_KD']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, feature_cols

def predict_for_systems(model, features, feature_cols, dbdq, dbdt):
    results = []
    
    for system_name, system_data in {**dbdq, **dbdt}.items():
        for _, solvent_row in system_data.iterrows():
            try:
                input_features = features.copy()
                input_features['System'] = system_name
                input_features['Composition'] = str(solvent_row['Composition'])
                
                for col in solvent_row.index:
                    if any(col.startswith(prefix) for prefix in ['%Vol', '%Mol', '%Mas']):
                        try:
                            input_features[col] = float(solvent_row[col]) if not pd.isna(solvent_row[col]) else 0.0
                        except:
                            input_features[col] = 0.0
                
                input_df = pd.DataFrame([input_features])
                
                for col in feature_cols:
                    if col not in input_df:
                        input_df[col] = 0.0
                
                log_kd = model.predict(input_df[feature_cols])[0]
                
                if -1 <= log_kd <= 1:
                    composition_list = []
                    for i in range(1, 5):
                        col = f'%Vol{i} - UP'
                        if col in solvent_row and not pd.isna(solvent_row[col]):
                            composition_list.append(f"{float(solvent_row[col]):.3f}")
                    
                    results.append({
                        "System": system_name,
                        "Composition": " / ".join(composition_list),
                        "Composition ID": str(solvent_row['Composition']),
                        "Predicted Log KD": f"{log_kd:.2f}"
                    })
            except Exception as e:
                if debug_mode:
                    st.warning(f"Skipping prediction for {system_name}-{solvent_row.get('Composition', '?')}: {str(e)}")
                continue
    
    return results

def main():
    st.markdown("""
    Entrez une chaîne SMILES pour obtenir des recommandations de systèmes de solvants avec un log KD prédit entre -1 et 1.
    """)
    
    with st.spinner("Chargement des données..."):
        kddb, dbdq, dbdt = load_data()
    
    if kddb is None or dbdq is None or dbdt is None:
        st.error("Échec du chargement des fichiers de données. Veuillez vérifier que les fichiers existent et sont des fichiers Excel valides.")
        return
    
    with st.spinner("Préparation des données d'entraînement..."):
        training_data = prepare_training_data(kddb, dbdq, dbdt)
        if debug_mode:
            show_debug_data(training_data, "Training Data")
    
    if training_data.empty:
        st.error("Aucune donnée d'entraînement valide n'a pu être préparée.")
        return
    
    with st.spinner("Entraînement du modèle..."):
        model, feature_cols = train_model(training_data)
    
    if model is None:
        st.error("Échec de l'entraînement du modèle.")
        return
    
    st.subheader("Entrée de la molécule")
    smiles = st.text_input("Entrez une chaîne SMILES", value="C1=CC(=CC=C1/C=C/C2=CC(=CC(=C2)O)O)O")
    
    if st.button("Trouver des systèmes de solvants appropriés"):
        if not smiles:
            st.warning("Veuillez entrer une chaîne SMILES")
            return
            
        with st.spinner("Calcul des caractéristiques moléculaires..."):
            features = extract_features(smiles)
        
        if features is None:
            st.error("Chaîne SMILES invalide.")
            return

        mol = Chem.MolFromSmiles(smiles)
        if mol:
            st.subheader("Structure moléculaire")
            img = Draw.MolToImage(mol, size=(400, 400))
            buf = BytesIO()
            img.save(buf, format="PNG")
            st.image(buf.getvalue(), width=200)

        with st.spinner("Prédiction des systèmes de solvants..."):
            results = predict_for_systems(model, features, feature_cols, dbdq, dbdt)

        if results:
            st.subheader("🔍 Systèmes de solvants prédits")
            df_results = pd.DataFrame(results)

            # ✅ Correction pour compatibilité avec pyarrow / streamlit
            df_results = df_results.astype({col: str for col in df_results.columns if df_results[col].dtype == 'object'})

            st.dataframe(
                df_results.sort_values("Predicted Log KD", key=lambda x: abs(x.astype(float)), ascending=True),
                use_container_width=True
            )
        else:
            st.warning("Aucun système de solvants prédit avec un log KD entre -1 et 1.")

if __name__ == "__main__":
    main()
