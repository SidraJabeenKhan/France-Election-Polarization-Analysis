#!/usr/bin/env python3
"""
France Election Polarization Analysis - Data Simulation Module
Generates synthetic social media data that mirrors real French electoral patterns.
Author: Sidra Jabeen Khan
Date: June 2026
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# French electoral hashtags and keywords
HASHTAGS = {
    'macron': ['#Macron', '#Macron2022', '#Macron2027', '#Ensemble', '#Renaissance', 
               '#AvecMacron', '#MacronPresident'],
    'lepen': ['#LePen', '#MarineLePen', '#RN', '#RassemblementNational', 
              '#LePen2022', '#LePen2027', '#VivementLePen'],
    'melenchon': ['#Melenchon', '#JLM', '#LFI', '#LaFranceInsoumise', 
                  '#Melenchon2022', '#UnionPopulaire', '#Nupes'],
    'center_right': ['#Pecresse', '#LR', '#LesRepublicains', '#ValeriePecresse'],
    'green': ['#Jadot', '#EELV', '#EuropeEcologie', '#YannickJadot'],
    'far_right': ['#Zemmour', '#Reconquete', '#EricZemmour'],
    'general': ['#Presidentielle2022', '#Presidentielle2027', '#Election2022', 
                '#Election2027', '#France', '#Republique', '#Democratie', 
                '#Vote', '#Citoyen']
}

# Political orientation mapping
ORIENTATIONS = {
    'macron': 'centrist',
    'lepen': 'far_right',
    'melenchon': 'far_left',
    'center_right': 'center_right',
    'green': 'green',
    'far_right': 'extreme_right',
    'general': 'neutral'
}

# French departments with approximate political leanings (simplified)
DEPARTMENTS = {
    'Paris': {'code': '75', 'region': 'Ile-de-France', 'leaning': 'left_urban', 'lat': 48.8566, 'lon': 2.3522},
    'Seine-Saint-Denis': {'code': '93', 'region': 'Ile-de-France', 'leaning': 'left_urban', 'lat': 48.9137, 'lon': 2.4244},
    'Bouches-du-Rhone': {'code': '13', 'region': 'Provence-Alpes-Cote d\'Azur', 'leaning': 'mixed_urban', 'lat': 43.2965, 'lon': 5.3698},
    'Nord': {'code': '59', 'region': 'Hauts-de-France', 'leaning': 'left_industrial', 'lat': 50.6292, 'lon': 3.0573},
    'Rhone': {'code': '69', 'region': 'Auvergne-Rhone-Alpes', 'leaning': 'mixed_urban', 'lat': 45.7640, 'lon': 4.8357},
    'Haute-Garonne': {'code': '31', 'region': 'Occitanie', 'leaning': 'left_urban', 'lat': 43.6047, 'lon': 1.4442},
    'Gironde': {'code': '33', 'region': 'Nouvelle-Aquitaine', 'leaning': 'mixed', 'lat': 44.8378, 'lon': -0.5792},
    'Loire-Atlantique': {'code': '44', 'region': 'Pays de la Loire', 'leaning': 'mixed', 'lat': 47.2184, 'lon': -1.5536},
    'Bas-Rhin': {'code': '67', 'region': 'Grand Est', 'leaning': 'center_right', 'lat': 48.5734, 'lon': 7.7528},
    'Calvados': {'code': '14', 'region': 'Normandie', 'leaning': 'mixed', 'lat': 49.1829, 'lon': -0.3707},
    'Finistere': {'code': '29', 'region': 'Bretagne', 'leaning': 'left', 'lat': 48.2020, 'lon': -4.0200},
    'Herault': {'code': '34', 'region': 'Occitanie', 'leaning': 'mixed', 'lat': 43.6117, 'lon': 3.8767},
    'Isere': {'code': '38', 'region': 'Auvergne-Rhone-Alpes', 'leaning': 'mixed', 'lat': 45.1885, 'lon': 5.7245},
    'Puy-de-Dome': {'code': '63', 'region': 'Auvergne-Rhone-Alpes', 'leaning': 'left', 'lat': 45.7719, 'lon': 3.0848},
    'Seine-Maritime': {'code': '76', 'region': 'Normandie', 'leaning': 'mixed', 'lat': 49.4432, 'lon': 1.0999},
    'Var': {'code': '83', 'region': 'Provence-Alpes-Cote d\'Azur', 'leaning': 'right', 'lat': 43.1242, 'lon': 6.1286},
    'Vaucluse': {'code': '84', 'region': 'Provence-Alpes-Cote d\'Azur', 'leaning': 'right', 'lat': 44.0562, 'lon': 5.0466},
    'Yvelines': {'code': '78', 'region': 'Ile-de-France', 'leaning': 'center_right', 'lat': 48.8049, 'lon': 2.1204},
    'Alpes-Maritimes': {'code': '06', 'region': 'Provence-Alpes-Cote d\'Azur', 'leaning': 'right', 'lat': 43.7102, 'lon': 7.2620},
    'Cote-d\'Or': {'code': '21', 'region': 'Bourgogne-Franche-Comte', 'leaning': 'mixed', 'lat': 47.3216, 'lon': 5.0415}
}

def generate_election_timeline(election_date, pre_months=3, post_months=1):
    """Generate timeline around election date."""
    start_date = election_date - timedelta(days=pre_months*30)
    end_date = election_date + timedelta(days=post_months*30)
    return pd.date_range(start=start_date, end=end_date, freq='H')

def assign_polarization_score(text, orientation):
    """
    Assign polarization score based on text content and political orientation.
    Higher scores indicate more polarized/radicalized language.
    """
    base_score = 0.3  # Neutral baseline
    
    # Adjust based on orientation extremity
    extremity = {
        'centrist': 0.0,
        'center_right': 0.1,
        'green': 0.1,
        'mixed': 0.0,
        'left': 0.15,
        'left_urban': 0.2,
        'left_industrial': 0.2,
        'right': 0.2,
        'far_right': 0.35,
        'far_left': 0.35,
        'extreme_right': 0.45,
        'neutral': 0.0
    }
    
    score = base_score + extremity.get(orientation, 0.0)
    
    # Add random variation
    score += np.random.normal(0, 0.1)
    
    # Clamp to 0-1 range
    return max(0.0, min(1.0, score))

def generate_synthetic_posts(n_posts=5000, election_date=None):
    """
    Generate synthetic social media posts for French electoral context.
    
    Parameters:
    -----------
    n_posts : int
        Number of posts to generate
    election_date : datetime
        Election date (default: April 24, 2022 for 2022 presidential runoff)
    """
    if election_date is None:
        election_date = datetime(2022, 4, 24)
    
    timeline = generate_election_timeline(election_date)
    
    posts = []
    for i in range(n_posts):
        # Select political camp
        camp = random.choices(
            list(HASHTAGS.keys()),
            weights=[25, 20, 20, 10, 5, 8, 12],  # Macron, LePen, Melenchon most common
            k=1
        )[0]
        
        # Select hashtags
        hashtags = random.sample(HASHTAGS[camp], k=random.randint(1, 3))
        hashtag_str = ' '.join(hashtags)
        
        # Select department (with some geographic bias toward camp strongholds)
        dept_name = random.choice(list(DEPARTMENTS.keys()))
        dept_info = DEPARTMENTS[dept_name]
        
        # Generate timestamp (more posts closer to election)
        days_to_election = (election_date - timeline[0]).days
        weights = [1 / (1 + abs((t - election_date).days)) for t in timeline]
        weights = np.array(weights) / sum(weights)
        timestamp = np.random.choice(timeline, p=weights)
        
        # Generate engagement metrics (correlated with polarization)
        base_engagement = random.randint(10, 500)
        
        # Generate text (simplified - in real project, would use more sophisticated generation)
        text_templates = {
            'macron': [
                "Ensemble pour la Republique! {hashtags}",
                "Le progres continue avec {hashtags}",
                "L'Europe, l'ecologie, l'economie - {hashtags}",
                "Pour une France unie et forte {hashtags}"
            ],
            'lepen': [
                "La France d'abord! {hashtags}",
                "Protegeons nos frontieres {hashtags}",
                "Pour le peuple francais {hashtags}",
                "Stop immigration {hashtags}"
            ],
            'melenchon': [
                "L'Union Populaire vaincra! {hashtags}",
                "Pour le partage des richesses {hashtags}",
                "Revolution citoyenne {hashtags}",
                "Contre les oligarchies {hashtags}"
            ],
            'center_right': [
                "La droite republicaine {hashtags}",
                "Pour des valeurs sures {hashtags}"
            ],
            'green': [
                "Pour la planete {hashtags}",
                "L'ecologie c'est maintenant {hashtags}"
            ],
            'far_right': [
                "Reconquete! {hashtags}",
                "La grande remigration {hashtags}"
            ],
            'general': [
                "Democratie en action {hashtags}",
                "Votez pour l'avenir {hashtags}",
                "Citoyens engages {hashtags}"
            ]
        }
        
        text = random.choice(text_templates[camp]).format(hashtags=hashtag_str)
        
        # Calculate polarization score
        polarization = assign_polarization_score(text, ORIENTATIONS[camp])
        
        # Engagement correlates with polarization (controversial posts get more engagement)
        engagement = int(base_engagement * (1 + polarization * 2))
        
        # Determine if post is retweet (30% chance)
        is_retweet = random.random() < 0.3
        
        # Original author (for retweets)
        if is_retweet:
            original_author = f"user_{random.randint(1000, 9999)}"
        else:
            original_author = None
        
        post = {
            'post_id': f"post_{i:06d}",
            'user_id': f"user_{random.randint(1000, 9999)}",
            'timestamp': timestamp,
            'text': text,
            'hashtags': hashtag_str,
            'camp': camp,
            'orientation': ORIENTATIONS[camp],
            'department': dept_name,
            'dept_code': dept_info['code'],
            'region': dept_info['region'],
            'latitude': dept_info['lat'] + np.random.normal(0, 0.1),
            'longitude': dept_info['lon'] + np.random.normal(0, 0.1),
            'polarization_score': round(polarization, 3),
            'engagement': engagement,
            'likes': int(engagement * 0.6),
            'retweets': int(engagement * 0.25),
            'replies': int(engagement * 0.15),
            'is_retweet': is_retweet,
            'original_author': original_author,
            'days_to_election': (election_date - timestamp).days
        }
        
        posts.append(post)
    
    df = pd.DataFrame(posts)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df

if __name__ == "__main__":
    # Generate dataset for 2022 election
    print("Generating synthetic French election social media dataset...")
    df = generate_synthetic_posts(n_posts=5000, election_date=datetime(2022, 4, 24))
    
    # Save to CSV
    df.to_csv('france_election_2022_simulated.csv', index=False, encoding='utf-8')
    print(f"Generated {len(df)} synthetic posts")
    print(f"Saved to france_election_2022_simulated.csv")
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nSample data:")
    print(df.head(3)[['timestamp', 'camp', 'orientation', 'department', 'polarization_score', 'engagement']])
