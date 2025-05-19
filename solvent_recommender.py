import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Draw
import numpy as np
from io import BytesIO
from PIL import Image

# Configuration
st.set_page_config(page_title="Solvent System Recommender", layout="wide")
st.title("🔬 Solvent System Recommender")

# Debug flag
debug_mode = st.checkbox("Enable debug mode", value=True)  # Forcé à True pour le débogage

def load_and_validate_data():
    try:
        kddb = pd.read_excel("KDDB.xlsx", sheet_name=None)
        dbdq = pd.read_excel("DBDQ.xlsx", sheet_name=None)
        dbdt = pd.read_excel("DBDT.xlsx", sheet_name=None)
        
        # Validation des données critiques
        for name, data in [('KDDB', kddb), ('DBDQ', dbdq), ('DBDT', dbdt)]:
            if not all(isinstance(sheet, pd.DataFrame) for sheet in data.values()):
                st.error(f"{name} contient des feuilles invalides")
                return None, None, None
                
        return kddb, dbdq, dbdt
    except Exception as e:
        st.error(f"Erreur de chargement: {str(e)}")
        return None, None, None

def prepare_features(row, solvent_data):
    """Crée un dictionnaire de caractéristiques complet"""
    features = {
        'MolWeight': Descriptors.MolWt(Chem.MolFromSmiles(row['SMILES'])),
        'LogP': Descriptors.MolLogP(Chem.MolFromSmiles(row['SMILES'])),
        'HBD': Lipinski.NumHDonors(Chem.MolFromSmiles(row['SMILES'])),
        'HBA': Lipinski.NumHAcceptors(Chem.MolFromSmiles(row['SMILES'])),
        'TPSA': Descriptors.TPSA(Chem.MolFromSmiles(row['SMILES'])),
        'RotatableBonds': Lipinski.NumRotatableBonds(Chem.MolFromSmiles(row['SMILES'])),
        'AromaticRings': Lipinski.NumAromaticRings(Chem.MolFromSmiles(row['SMILES'])),
        'HeavyAtoms': Lipinski.HeavyAtomCount(Chem.MolFromSmiles(row['SMILES'])),
        'Log_KD': float(row['Log KD'])
    }
    
    # Ajout des caractéristiques de composition avec vérification rigoureuse
    composition_features = {}
    for i in range(1, 5):  # Pour 4 solvants max
        for prefix in ['%Vol', '%Mol', '%Mas']:
            col = f"{prefix}{i} - UP"
            if col in solvent_data:
                try:
                    composition_features[f"{prefix}{i}"] = float(solvent_data[col])
                except:
                    composition_features[f"{prefix}{i}"] = 0.0
    
    return {**features, **composition_features}

def train_and_validate_model(X, y):
    """Entraîne le modèle avec validation croisée"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=25,
        min_samples_split=4,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Validation approfondie
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    metrics = {
        'train_r2': model.score(X_train, y_train),
        'test_r2': model.score(X_test, y_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred))
    }
    
    if debug_mode:
        st.write("🔍 Validation du modèle:")
        st.write(f"R² (train): {metrics['train_r2']:.3f} | R² (test): {metrics['test_r2']:.3f}")
        st.write(f"RMSE (train): {metrics['train_rmse']:.3f} | RMSE (test): {metrics['test_rmse']:.3f}")
        
        # Analyse des prédictions
        pred_analysis = pd.DataFrame({
            'Actual': y_test,
            'Predicted': test_pred,
            'Difference': np.abs(y_test - test_pred)
        })
        st.write("Échantillon des prédictions:", pred_analysis.head(10))
        
        # Importance des caractéristiques
        importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        st.write("Top 10 caractéristiques importantes:", importances.head(10))
    
    return model, metrics

def main():
    st.markdown("""
    ## Système de Recommandation de Solvants
    Entrez un SMILES pour obtenir des recommandations avec un log KD prédit entre -1 et 1.
    """)
    
    # Chargement des données
    with st.spinner("Chargement et validation des données..."):
        kddb, dbdq, dbdt = load_and_validate_data()
        if None in (kddb, dbdq, dbdt):
            return
        
        # Préparation des données
        data = []
        for sheet_name, sheet_data in kddb.items():
            for _, row in sheet_data.iterrows():
                try:
                    if pd.isna(row['SMILES']) or pd.isna(row['Log KD']) or pd.isna(row['System']):
                        continue
                        
                    # Recherche des données de solvant correspondantes
                    solvent_data = None
                    for db in [dbdq, dbdt]:
                        if row['System'] in db:
                            match = db[row['System']][db[row['System']]['Composition'].astype(str) == str(row['Composition'])]
                            if not match.empty:
                                solvent_data = match.iloc[0].to_dict()
                                break
                    
                    if solvent_data:
                        features = prepare_features(row, solvent_data)
                        data.append(features)
                except Exception as e:
                    if debug_mode:
                        st.warning(f"Erreur ligne {_}: {str(e)}")
        
        df = pd.DataFrame(data)
        
        if debug_mode:
            st.write("📊 Statistiques des données:")
            st.write(f"Nombre total d'échantillons: {len(df)}")
            st.write("Distribution du Log KD:", df['Log_KD'].describe())
            
            # Vérification des valeurs de composition
            comp_features = [c for c in df.columns if any(c.startswith(p) for p in ['%Vol', '%Mol', '%Mas'])]
            if comp_features:
                st.write("Valeurs de composition:", df[comp_features].describe())

        if df.empty:
            st.error("Aucune donnée valide n'a pu être préparée.")
            return
        
        # Entraînement du modèle
        X = df.drop(['Log_KD'], axis=1)
        y = df['Log_KD']
        
        model, metrics = train_and_validate_model(X, y)
        
        if metrics['test_r2'] < 0.3:
            st.error("Le modèle n'a pas appris correctement (R² trop faible)")
            return

    # Interface utilisateur
    smiles = st.text_input("SMILES", value="C1=CC(=CC=C1/C=C/C2=CC(=CC(=C2)O)O)O")
    
    if st.button("Recommander des systèmes"):
        if not smiles:
            st.warning("Veuillez entrer un SMILES valide")
            return
            
        try:
            # Calcul des caractéristiques moléculaires
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                st.error("SMILES invalide")
                return
                
            features = {
                'MolWeight': Descriptors.MolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'HBD': Lipinski.NumHDonors(mol),
                'HBA': Lipinski.NumHAcceptors(mol),
                'TPSA': Descriptors.TPSA(mol),
                'RotatableBonds': Lipinski.NumRotatableBonds(mol),
                'AromaticRings': Lipinski.NumAromaticRings(mol),
                'HeavyAtoms': Lipinski.HeavyAtomCount(mol)
            }
            
            # Affichage de la structure
            st.subheader("Structure Moléculaire")
            img = Draw.MolToImage(mol, size=(400, 400))
            st.image(img, width=300)
            
            # Prédictions pour tous les systèmes
            results = []
            for system_name, system_data in {**dbdq, **dbdt}.items():
                for _, solvent_row in system_data.iterrows():
                    try:
                        # Préparation des caractéristiques de composition
                        input_features = features.copy()
                        for i in range(1, 5):
                            for prefix in ['%Vol', '%Mol', '%Mas']:
                                col = f"{prefix}{i} - UP"
                                if col in solvent_row:
                                    input_features[f"{prefix}{i}"] = float(solvent_row[col]) if not pd.isna(solvent_row[col]) else 0.0
                        
                        # Création du DataFrame d'entrée
                        input_df = pd.DataFrame([input_features])[X.columns]
                        
                        # Prédiction
                        log_kd = model.predict(input_df)[0]
                        
                        if debug_mode:
                            st.write(f"Système: {system_name}-{solvent_row['Composition']}, KD prédit: {log_kd:.3f}")
                        
                        if -1 <= log_kd <= 1:
                            # Formatage des résultats
                            solvents = []
                            composition = []
                            for i in range(1, 5):
                                name_col = f"Solvent {i}"
                                vol_col = f"%Vol{i} - UP"
                                if name_col in solvent_row:
                                    solvents.append(str(solvent_row[name_col]))
                                if vol_col in solvent_row:
                                    composition.append(f"{float(solvent_row[vol_col]):.1f}%")
                            
                            results.append({
                                'Système': system_name,
                                'Solvants': " / ".join(solvents),
                                'Composition': " / ".join(composition),
                                'Log_KD_prédit': f"{log_kd:.2f}"
                            })
                            
                    except Exception as e:
                        if debug_mode:
                            st.warning(f"Erreur sur {system_name}-{solvent_row['Composition']}: {str(e)}")
            
            if results:
                df_results = pd.DataFrame(results)
                df_results['KD_abs'] = df_results['Log_KD_prédit'].astype(float).abs()
                df_results = df_results.sort_values('KD_abs')
                
                st.subheader("🎯 Systèmes Recommandés")
                st.dataframe(
                    df_results.drop('KD_abs', axis=1),
                    column_config={
                        "Log_KD_prédit": st.column_config.NumberColumn(
                            format="%.2f",
                            help="Valeur prédite du log KD (idéal entre -1 et 1)"
                        )
                    },
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.warning("Aucun système ne satisfait -1 ≤ log KD ≤ 1")
                
        except Exception as e:
            st.error(f"Erreur: {str(e)}")

if __name__ == "__main__":
    main()
