#!/usr/bin/env python3
"""
France Election Polarization Analysis - Geographic Visualization Module
Demonstrates choropleth mapping, regional polarization indices, and territorial analysis.
Author: Sidra Jabeen Khan
Date: June 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_regional_polarization(df):
    """
    Calculate polarization indices by French department and region.
    """
    print("=== CALCULATING REGIONAL POLARIZATION INDICES ===")
    
    # Department-level aggregation
    dept_stats = df.groupby('department').agg({
        'polarization_score': ['mean', 'std', 'count'],
        'engagement': 'mean',
        'narrative_class': lambda x: (x == 'radicalizing').mean(),
        'latitude': 'first',
        'longitude': 'first',
        'region': 'first',
        'dept_code': 'first'
    }).reset_index()
    
    dept_stats.columns = ['department', 'avg_polarization', 'polarization_std', 
                         'n_posts', 'avg_engagement', 'radicalizing_ratio',
                         'latitude', 'longitude', 'region', 'dept_code']
    
    # Region-level aggregation
    region_stats = df.groupby('region').agg({
        'polarization_score': 'mean',
        'engagement': 'mean',
        'narrative_class': lambda x: (x == 'radicalizing').mean()
    }).reset_index()
    region_stats.columns = ['region', 'avg_polarization', 'avg_engagement', 'radicalizing_ratio']
    
    print(f"\nDepartment-level statistics for {len(dept_stats)} departments:")
    print(dept_stats[['department', 'avg_polarization', 'radicalizing_ratio', 'n_posts']].head(10))
    
    print(f"\nRegion-level statistics:")
    print(region_stats.sort_values('avg_polarization', ascending=False))
    
    return dept_stats, region_stats

def create_choropleth_map(dept_stats):
    """
    Create choropleth map showing polarization by department.
    Uses matplotlib scatter plot as proxy for actual choropleth.
    """
    print("\n=== CREATING CHOROPLETH VISUALIZATION ===")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. Scatter plot map (polarization intensity)
    scatter = axes[0].scatter(dept_stats['longitude'], dept_stats['latitude'],
                             c=dept_stats['avg_polarization'], 
                             s=dept_stats['n_posts'] * 2,
                             cmap='RdYlBu_r', alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Add department labels
    for _, row in dept_stats.iterrows():
        axes[0].annotate(row['department'], 
                        (row['longitude'], row['latitude']),
                        fontsize=8, alpha=0.8)
    
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    axes[0].set_title('Polarization Index by French Department\n(Size = Post volume, Color = Polarization)')
    plt.colorbar(scatter, ax=axes[0], label='Polarization Index')
    
    # 2. Radicalizing ratio map
    scatter2 = axes[1].scatter(dept_stats['longitude'], dept_stats['latitude'],
                              c=dept_stats['radicalizing_ratio'], 
                              s=dept_stats['n_posts'] * 2,
                              cmap='Reds', alpha=0.7, edgecolors='black', linewidth=0.5)
    
    for _, row in dept_stats.iterrows():
        axes[1].annotate(row['department'], 
                        (row['longitude'], row['latitude']),
                        fontsize=8, alpha=0.8)
    
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    axes[1].set_title('Radicalizing Content Ratio by Department\n(Size = Post volume, Color = Radicalizing ratio)')
    plt.colorbar(scatter2, ax=axes[1], label='Radicalizing Ratio')
    
    plt.tight_layout()
    plt.savefig('geographic_analysis_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nVisualization saved as 'geographic_analysis_results.png'")

def analyze_urban_rural_divide(df):
    """
    Analyze polarization differences between urban and rural departments.
    """
    print("\n=== URBAN-RURAL POLARIZATION ANALYSIS ===")
    
    # Classify departments (simplified classification)
    urban_depts = ['Paris', 'Seine-Saint-Denis', 'Bouches-du-Rhone', 'Rhone', 
                   'Haute-Garonne', 'Loire-Atlantique']
    
    df['urban_rural'] = df['department'].apply(
        lambda x: 'urban' if x in urban_depts else 'rural'
    )
    
    urban_rural_stats = df.groupby('urban_rural').agg({
        'polarization_score': 'mean',
        'engagement': 'mean',
        'narrative_class': lambda x: (x == 'radicalizing').mean()
    }).reset_index()
    urban_rural_stats.columns = ['urban_rural', 'avg_polarization', 'avg_engagement', 'radicalizing_ratio']
    
    print(urban_rural_stats)
    
    # Statistical test (simplified)
    urban_polarization = df[df['urban_rural'] == 'urban']['polarization_score']
    rural_polarization = df[df['urban_rural'] == 'rural']['polarization_score']
    
    print(f"\nUrban polarization: {urban_polarization.mean():.3f} (n={len(urban_polarization)})")
    print(f"Rural polarization: {rural_polarization.mean():.3f} (n={len(rural_polarization)})")
    print(f"Difference: {urban_polarization.mean() - rural_polarization.mean():.3f}")
    
    return urban_rural_stats

def create_regional_comparison(region_stats):
    """
    Create bar chart comparing polarization across regions.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    region_stats_sorted = region_stats.sort_values('avg_polarization', ascending=True)
    
    colors = ['green' if x < 0.4 else 'orange' if x < 0.6 else 'red' 
              for x in region_stats_sorted['avg_polarization']]
    
    ax.barh(region_stats_sorted['region'], region_stats_sorted['avg_polarization'], 
           color=colors, alpha=0.7, edgecolor='black')
    
    ax.axvline(x=0.4, color='green', linestyle='--', alpha=0.5, label='Low polarization')
    ax.axvline(x=0.6, color='red', linestyle='--', alpha=0.5, label='High polarization')
    
    ax.set_xlabel('Average Polarization Index')
    ax.set_title('Polarization Index by French Region')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('regional_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nRegional comparison saved as 'regional_comparison.png'")

if __name__ == "__main__":
    # Load processed data
    print("Loading processed data...")
    df = pd.read_csv('france_election_2022_processed.csv')
    
    # Calculate regional statistics
    dept_stats, region_stats = calculate_regional_polarization(df)
    
    # Create visualizations
    create_choropleth_map(dept_stats)
    
    # Urban-rural analysis
    urban_rural_stats = analyze_urban_rural_divide(df)
    
    # Regional comparison
    create_regional_comparison(region_stats)
    
    print("\n=== GEOGRAPHIC ANALYSIS SUMMARY ===")
    print(f"Departments analyzed: {len(dept_stats)}")
    print(f"Regions analyzed: {len(region_stats)}")
    print(f"Most polarized region: {region_stats.loc[region_stats['avg_polarization'].idxmax(), 'region']}")
    print(f"Least polarized region: {region_stats.loc[region_stats['avg_polarization'].idxmin(), 'region']}")
