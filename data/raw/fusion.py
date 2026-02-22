import pandas as pd

fichiers = [
    'requetes311_2014-2016.csv',
    'requetes311_2016-2018.csv',
    'requetes311_2017-2018.csv',
    'requetes311_2019-2021.csv',
    'requetes311_today.csv'
]

categories = [
    'Feux de circulation - Entretien', 'Signalisation - Circulation - Études',
    'Collecte de déchets', 'Bac roulant', "Fermeture d'une conduite d'eau - Urgence",
    'Collecte des encombrants', 'Info-Remorquage', 'Neige - Remorquage',
    'Opération déneigement', 'Nid-de-poule', "Fermeture d'entrée d'eau",
    'Permis - Divers', 'Info-travaux - Service central', 'Collecte des matières recyclables', 
    'Branche tombée', 'Danger potentiel - Arbre', "Info - Collecte d'objets encombrants",
    'Collecte des matières organiques', 'Permis - Neige - Domaine public',
    'Info-travaux - Arrondissement', 'Intervention stationnement', 'Stationnement municipal',
    'Carte Info-Neige', 'Trottoir glissant', 'Chaussée glissante',
    'Piste cyclable déneigement', 'Pavage - Réparation', 'Trottoir ou bordure - Réparation',
    "Fuite d'eau", 'Égout - Crue des eaux', 'Signalisation endommagée Vm',
    '*Signalisation manquante ou endommagée', '*Circulation - Travaux majeurs',
    '*Entrave - Piste cyclable', 'Débris sur la voie publique', 'Arbre tombé'
]

df_clean_list = []

for fichier in fichiers:
    df = pd.read_csv(fichier, low_memory=False)
    
    if 'ACTI_NOM' in df.columns:
        df_filtered = df[df['ACTI_NOM'].isin(categories)].copy()
        
        if 'LOC_LAT' in df_filtered.columns and 'LOC_LONG' in df_filtered.columns:
            df_filtered = df_filtered.dropna(subset=['LOC_LAT', 'LOC_LONG'])
            
        df_clean_list.append(df_filtered)

df_final = pd.concat(df_clean_list, ignore_index=True)
df_final = df_final.drop_duplicates()
df_final['DATE_DERNIER_STATUT'] = pd.to_datetime(df_final['DATE_DERNIER_STATUT'], errors='coerce')
date_min = df_final['DATE_DERNIER_STATUT'].min()
print(date_min)
df_final.to_csv('requetes_311.csv', index=False)