import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import PyPDF2
import webbrowser
import os
from io import BytesIO
import plotly.graph_objects as go
from matplotlib.animation import FuncAnimation
from matplotlib import patches
import base64

# Configuration de la page
st.set_page_config(page_title="Quaterco", layout="wide", page_icon=":chart_with_upwards_trend:")

# Style CSS personnalisé
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        height: 3em;
        width: 100%;
        border-radius: 5px;
        font-size: 16px;
    }
    .stFileUploader>div>div>button {
        background-color: #4CAF50;
        color: white;
    }
    .css-1aumxhk {
        background-color: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)

# Titre de l'application
st.title("Quaterco - Outil d'analyse des paramètres de solubilité et diagrammes de phase")

# Navigation entre les modules
modules = ["Paramètres de Solubilité de Hansen", "Diagramme de Phase Ternaire", "Diagramme de Phase Quaternaire"]
selected_module = st.sidebar.radio("Sélectionnez un module", modules)

# Fonction pour télécharger un fichier PDF
def get_pdf_download_link(pdf_path, text="Télécharger le PDF"):
    with open(pdf_path, "rb") as f:
        pdf_data = f.read()
    b64 = base64.b64encode(pdf_data).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{os.path.basename(pdf_path)}">{text}</a>'
    return href

# Module Paramètres de Solubilité de Hansen
if selected_module == "Paramètres de Solubilité de Hansen":
    st.header("Paramètres de Solubilité de Hansen")
    
    # Upload du fichier Excel
    uploaded_file = st.file_uploader("Téléversez votre fichier Excel pour les Paramètres de Hansen", type=["xlsx"])
    
    if uploaded_file is not None:
        try:
            data = pd.read_excel(uploaded_file)
            
            # Vérification des colonnes nécessaires
            required_columns = ['δD', 'δP', 'δH', 'Type', 'Compounds', 'CAS', 'R0']
            if not all(col in data.columns for col in required_columns):
                st.error(f"Le fichier Excel doit contenir les colonnes suivantes: {', '.join(required_columns)}")
            else:
                # Extraction des données
                x = data['δD'].astype(float)
                y = data['δP'].astype(float)
                z = data['δH'].astype(float)
                names = data['Compounds']
                types = data['Type']
                CAS = data['CAS']
                radii = data['R0'].astype(float)
                
                # Couleurs et légendes
                colors = ['red', 'green', 'blue']
                type_labels = ['Non-Green', 'Green', 'Simulated']
                
                # Création du graphique 3D avec Plotly
                fig = go.Figure()
                
                # Ajout des points avec des couleurs selon le type
                for i in range(len(x)):
                    fig.add_trace(go.Scatter3d(
                        x=[x[i]],
                        y=[y[i]],
                        z=[z[i]],
                        mode='markers',
                        marker=dict(
                            size=5,
                            color=colors[types[i]],
                            opacity=0.8
                        ),
                        name=names[i],
                        text=f"Nom: {names[i]}<br>CAS: {CAS[i]}<br>δD: {x[i]}<br>δP: {y[i]}<br>δH: {z[i]}<br>R0: {radii[i]}",
                        hoverinfo='text'
                    ))
                
                # Mise en forme du graphique
                fig.update_layout(
                    scene=dict(
                        xaxis_title='δD',
                        yaxis_title='δP',
                        zaxis_title='δH',
                    ),
                    margin=dict(l=0, r=0, b=0, t=30),
                    height=700
                )
                
                # Affichage du graphique
                st.plotly_chart(fig, use_container_width=True)
                
                # Légende
                legend_html = "<div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 20px;'>"
                legend_html += "<h4 style='margin-top:0;'>Légende:</h4>"
                for i, label in enumerate(type_labels):
                    legend_html += f"<span style='color:{colors[i]}; font-weight:bold;'>■</span> {label}<br>"
                legend_html += "<p><b>Interactions:</b><br>"
                legend_html += "- Passez la souris sur un point pour voir les détails<br>"
                legend_html += "- Utilisez les outils de navigation en haut à droite pour zoomer/pivoter</p>"
                legend_html += "</div>"
                
                st.markdown(legend_html, unsafe_allow_html=True)
                
                # Recherche de composé
                st.subheader("Recherche de composé")
                compound_name = st.text_input("Entrez le nom d'un composé pour le mettre en évidence:")
                
                if compound_name and compound_name in names.values:
                    idx = names[names == compound_name].index[0]
                    fig.add_trace(go.Scatter3d(
                        x=[x[idx]],
                        y=[y[idx]],
                        z=[z[idx]],
                        mode='markers',
                        marker=dict(
                            size=10,
                            color='yellow',
                            opacity=1
                        ),
                        name=f"Selected: {compound_name}"
                    ))
                    
                    # Affichage des informations du composé
                    st.success(f"Composé trouvé: {compound_name}")
                    st.write(f"**CAS:** {CAS[idx]}")
                    st.write(f"**Coordonnées:** δD={x[idx]}, δP={y[idx]}, δH={z[idx]}")
                    st.write(f"**Rayon d'interaction (R0):** {radii[idx]}")
                    
                    # Bouton pour ouvrir dans PubChem
                    if st.button(f"Rechercher {compound_name} sur PubChem"):
                        url = f"https://pubchem.ncbi.nlm.nih.gov/#query={CAS[idx]}"
                        webbrowser.open_new_tab(url)
                    
                    # Affichage du graphique mis à jour
                    st.plotly_chart(fig, use_container_width=True)
                
                elif compound_name:
                    st.warning("Composé non trouvé dans le fichier.")
                
                # Visualisation de la sphère d'interaction
                st.subheader("Visualisation de la sphère d'interaction")
                sphere_compound = st.selectbox("Sélectionnez un composé pour visualiser sa sphère d'interaction", [""] + list(names.unique()))
                
                if sphere_compound:
                    idx = names[names == sphere_compound].index[0]
                    center = (x[idx], y[idx], z[idx])
                    radius = radii[idx]
                    
                    # Création de la sphère
                    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                    sphere_x = center[0] + radius * np.cos(u) * np.sin(v)
                    sphere_y = center[1] + radius * np.sin(u) * np.sin(v)
                    sphere_z = center[2] + radius * np.cos(v)
                    
                    # Ajout de la sphère au graphique
                    fig.add_trace(go.Surface(
                        x=sphere_x,
                        y=sphere_y,
                        z=sphere_z,
                        colorscale=[[0, 'rgba(255, 255, 0, 0.1)'], [1, 'rgba(255, 255, 0, 0.1)']],
                        showscale=False,
                        name=f"Sphère d'interaction: {sphere_compound}"
                    ))
                    
                    # Affichage du graphique avec la sphère
                    st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Une erreur s'est produite lors du traitement du fichier: {str(e)}")

# Module Diagramme de Phase Ternaire
elif selected_module == "Diagramme de Phase Ternaire":
    st.header("Diagramme de Phase Ternaire")
    
    # Upload du fichier Excel
    uploaded_file = st.file_uploader("Téléversez votre fichier Excel pour le Diagramme Ternaire", type=["xlsx"])
    
    if uploaded_file is not None:
        try:
            data = pd.read_excel(uploaded_file)
            
            # Vérification des colonnes nécessaires
            required_columns = ['V1', 'V2', "V1'", "V2'"]
            if not all(col in data.columns for col in required_columns):
                st.error(f"Le fichier Excel doit contenir les colonnes suivantes: {', '.join(required_columns)}")
            else:
                # Extraction des données
                x = data['V1']
                y = data['V2']
                x_prime = data["V1'"]
                y_prime = data["V2'"]
                
                # Création du graphique
                fig, ax = plt.subplots(figsize=(8, 8))
                
                # Dessiner chaque ligne avec une couleur aléatoire
                for i in range(len(x)):
                    color = np.random.rand(3,)
                    ax.plot([x[i], x_prime[i]], [y[i], y_prime[i]], color=color, marker='o', markersize=5)
                
                # Coordonnées des sommets du triangle
                triangle_vertices = np.array([[0, 0], [1, 0], [0, 1], [0, 0]])
                
                # Tracer le triangle en noir
                ax.plot(triangle_vertices[:, 0], triangle_vertices[:, 1], color='black')
                
                # Ajouter une grille fine
                ax.set_xticks(np.arange(0, 1.05, 0.05))
                ax.set_yticks(np.arange(0, 1.05, 0.05))
                ax.grid(which='both', linestyle='-', linewidth=0.5)
                
                # Labels des axes
                ax.set_xlabel('V1' + (' ' + str(data.iloc[0, 7]) if len(data.columns) > 7 else ''))
                ax.set_ylabel('V2' + (' ' + str(data.iloc[0, 8]) if len(data.columns) > 8 else ''))
                
                # Titre
                if len(data.columns) > 9:
                    title_string = f"{data.iloc[0, 7]} / {data.iloc[0, 8]} / {data.iloc[0, 9]}"
                    ax.set_title(title_string)
                
                # Affichage dans Streamlit
                st.pyplot(fig)
                
                # Téléchargement du graphique
                buf = BytesIO()
                fig.savefig(buf, format="png", dpi=300)
                st.download_button(
                    label="Télécharger le diagramme",
                    data=buf.getvalue(),
                    file_name="ternary_phase_diagram.png",
                    mime="image/png"
                )
        
        except Exception as e:
            st.error(f"Une erreur s'est produite lors du traitement du fichier: {str(e)}")

# Module Diagramme de Phase Quaternaire
elif selected_module == "Diagramme de Phase Quaternaire":
    st.header("Diagramme de Phase Quaternaire")
    
    # Upload du fichier Excel
    uploaded_file = st.file_uploader("Téléversez votre fichier Excel pour le Diagramme Quaternaire", type=["xlsx"])
    
    if uploaded_file is not None:
        try:
            data = pd.read_excel(uploaded_file)
            
            # Vérification des colonnes nécessaires
            required_columns = ['V1', 'V2', 'V3', "V1'", "V2'", "V3'"]
            if not all(col in data.columns for col in required_columns):
                st.error(f"Le fichier Excel doit contenir les colonnes suivantes: {', '.join(required_columns)}")
            else:
                # Extraction des données
                x = data['V1']
                y = data['V2']
                z = data['V3']
                x_prime = data["V1'"]
                y_prime = data["V2'"]
                z_prime = data["V3'"]
                
                # Noms si disponibles
                names = data['Nom'] if 'Nom' in data.columns else [None] * len(x)
                
                # Création du graphique 3D avec Plotly
                fig = go.Figure()
                
                # Ajout des lignes entre les points
                for i in range(len(x)):
                    fig.add_trace(go.Scatter3d(
                        x=[x[i], x_prime[i]],
                        y=[y[i], y_prime[i]],
                        z=[z[i], z_prime[i]],
                        mode='lines+markers',
                        marker=dict(size=5, color=np.random.rand(3,).tolist()),
                        line=dict(width=2, color=np.random.rand(3,).tolist()),
                        name=names[i] if names[i] else f"Point {i+1}"
                    ))
                
                # Ajout de la pyramide
                pyramid_vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
                pyramid_edges = [
                    [0, 1], [0, 2], [0, 3],
                    [1, 2], [1, 3],
                    [2, 3]
                ]
                
                for edge in pyramid_edges:
                    fig.add_trace(go.Scatter3d(
                        x=[pyramid_vertices[edge[0]][0], pyramid_vertices[edge[1]][0]],
                        y=[pyramid_vertices[edge[0]][1], pyramid_vertices[edge[1]][1]],
                        z=[pyramid_vertices[edge[0]][2], pyramid_vertices[edge[1]][2]],
                        mode='lines',
                        line=dict(color='gray', width=2),
                        showlegend=False
                    ))
                
                # Mise en forme du graphique
                fig.update_layout(
                    scene=dict(
                        xaxis_title='V1' + (' ' + str(data.iloc[0, 9]) if len(data.columns) > 9 else ''),
                        yaxis_title='V2' + (' ' + str(data.iloc[0, 10]) if len(data.columns) > 10 else ''),
                        zaxis_title='V3' + (' ' + str(data.iloc[0, 11]) if len(data.columns) > 11 else ''),
                    ),
                    margin=dict(l=0, r=0, b=0, t=30),
                    height=700
                )
                
                # Titre
                if len(data.columns) > 12:
                    title_string = f"{data.iloc[0, 9]} / {data.iloc[0, 10]} / {data.iloc[0, 11]} / {data.iloc[0, 12]}"
                    fig.update_layout(title=title_string)
                
                # Affichage dans Streamlit
                st.plotly_chart(fig, use_container_width=True)
                
                # Animation de rotation
                if st.checkbox("Afficher l'animation de rotation"):
                    st.warning("L'animation peut prendre quelques secondes à se charger...")
                    
                    # Création d'une figure matplotlib pour l'animation
                    mfig = plt.figure(figsize=(8, 8))
                    max = mfig.add_subplot(111, projection='3d')
                    
                    # Ajout des lignes
                    for i in range(len(x)):
                        max.plot([x[i], x_prime[i]], [y[i], y_prime[i]], [z[i], z_prime[i]], 
                                color=np.random.rand(3,), marker='o', markersize=5)
                    
                    # Ajout de la pyramide
                    for edge in pyramid_edges:
                        max.plot([pyramid_vertices[edge[0]][0], pyramid_vertices[edge[1]][0]],
                                [pyramid_vertices[edge[0]][1], pyramid_vertices[edge[1]][1]],
                                [pyramid_vertices[edge[0]][2], pyramid_vertices[edge[1]][2]],
                                color='gray')
                    
                    # Fonction d'animation
                    def update(frame):
                        max.view_init(elev=20, azim=frame)
                        return max,
                    
                    # Création de l'animation
                    ani = FuncAnimation(mfig, update, frames=np.arange(0, 360, 2), interval=50, blit=False)
                    
                    # Affichage dans Streamlit
                    st.pyplot(mfig)
                    
                    # Sauvegarde et affichage du GIF
                    gif_path = "quaternary_rotation.gif"
                    ani.save(gif_path, writer='pillow', fps=20, dpi=100)
                    
                    with open(gif_path, "rb") as f:
                        gif_data = f.read()
                    
                    st.image(gif_data, caption="Animation de rotation du diagramme quaternaire")
                    
                    # Bouton de téléchargement
                    st.download_button(
                        label="Télécharger l'animation",
                        data=gif_data,
                        file_name="quaternary_rotation.gif",
                        mime="image/gif"
                    )
        
        except Exception as e:
            st.error(f"Une erreur s'est produite lors du traitement du fichier: {str(e)}")

# Pied de page
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: gray; font-size: 0.9em;">
    <p>Quaterco - Outil d'analyse des paramètres de solubilité et diagrammes de phase</p>
    <p>Développé avec Streamlit, Python, Matplotlib et Plotly</p>
    </div>
    """, unsafe_allow_html=True)
