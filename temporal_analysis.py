#!/usr/bin/env python3
"""
France Election Polarization Analysis - Temporal Analysis Module
Demonstrates time series analysis, event detection, and feedback loop identification.
Author: Sidra Jabeen Khan
Date: June 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def calculate_polarization_index(df, window='6H'):
    """
    Calculate polarization index over time using rolling windows.
    
    Polarization Index = 1 - (proportion of neutral/dialogue-oriented posts)
    Higher values indicate more polarized discourse.
    """
    print("=== CALCULATING POLARIZATION INDEX ===")
    
    # Set timestamp as index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_time = df.set_index('timestamp').sort_index()
    
    # Calculate polarization index in rolling windows
    polarization_series = []
    
    # Resample to regular intervals
    for timestamp, group in df_time.resample(window):
        if len(group) > 0:
            # Count narrative types
            narrative_counts = group['narrative_class'].value_counts(normalize=True)
            
            # Polarization = 1 - (dialogue_oriented + mobilizing proportion)
            # These are considered less polarizing
            neutral_prop = narrative_counts.get('dialogue_oriented', 0) + \
                          narrative_counts.get('mobilizing', 0) * 0.5
            
            polarization = 1 - neutral_prop
            
            # Weight by engagement
            avg_engagement = group['engagement'].mean()
            
            polarization_series.append({
                'timestamp': timestamp,
                'polarization_index': polarization,
                'avg_engagement': avg_engagement,
                'n_posts': len(group),
                'radicalizing_ratio': (group['narrative_class'] == 'radicalizing').mean(),
                'polarizing_ratio': (group['narrative_class'] == 'polarizing').mean(),
                'dialogue_ratio': (group['narrative_class'] == 'dialogue_oriented').mean()
            })
    
    polarization_df = pd.DataFrame(polarization_series)
    polarization_df = polarization_df.set_index('timestamp')
    
    print(f"Calculated polarization index for {len(polarization_df)} time windows")
    print(f"Average polarization: {polarization_df['polarization_index'].mean():.3f}")
    print(f"Max polarization: {polarization_df['polarization_index'].max():.3f}")
    print(f"Min polarization: {polarization_df['polarization_index'].min():.3f}")
    
    return polarization_df

def detect_escalation_events(polarization_df, threshold=0.7, min_duration=3):
    """
    Detect escalation events where polarization exceeds threshold for minimum duration.
    """
    print("\n=== DETECTING ESCALATION EVENTS ===")
    
    # Identify periods above threshold
    above_threshold = polarization_df['polarization_index'] > threshold
    
    # Find contiguous periods
    events = []
    in_event = False
    event_start = None
    
    for timestamp, is_above in above_threshold.items():
        if is_above and not in_event:
            in_event = True
            event_start = timestamp
        elif not is_above and in_event:
            in_event = False
            event_end = timestamp
            duration = (event_end - event_start).total_seconds() / 3600  # hours
            
            if duration >= min_duration:
                event_data = polarization_df.loc[event_start:event_end]
                events.append({
                    'start': event_start,
                    'end': event_end,
                    'duration_hours': duration,
                    'max_polarization': event_data['polarization_index'].max(),
                    'avg_polarization': event_data['polarization_index'].mean(),
                    'avg_engagement': event_data['avg_engagement'].mean()
                })
    
    # Handle case where event continues to end of data
    if in_event:
        event_end = polarization_df.index[-1]
        duration = (event_end - event_start).total_seconds() / 3600
        if duration >= min_duration:
            event_data = polarization_df.loc[event_start:event_end]
            events.append({
                'start': event_start,
                'end': event_end,
                'duration_hours': duration,
                'max_polarization': event_data['polarization_index'].max(),
                'avg_polarization': event_data['polarization_index'].mean(),
                'avg_engagement': event_data['avg_engagement'].mean()
            })
    
    print(f"Detected {len(events)} escalation events")
    for i, event in enumerate(events, 1):
        print(f"Event {i}: {event['start']} to {event['end']} "
              f"({event['duration_hours']:.1f}h, max_polarization={event['max_polarization']:.3f})")
    
    return events

def identify_feedback_loops(polarization_df, sentiment_col='avg_engagement', lag_hours=12):
    """
    Identify feedback loops between polarization and engagement.
    
    Theory: High polarization -> High engagement -> More visibility -> More polarization
    """
    print("\n=== IDENTIFYING FEEDBACK LOOPS ===")
    
    # Calculate lagged correlations
    polarization = polarization_df['polarization_index']
    engagement = polarization_df['avg_engagement']
    
    # Correlation at lag 0
    corr_0 = polarization.corr(engagement)
    print(f"Correlation (polarization <-> engagement, same time): {corr_0:.3f}")
    
    # Correlation with lag (polarization leads engagement)
    engagement_lagged = engagement.shift(-lag_hours//6)  # adjust for 6H windows
    corr_lag = polarization.corr(engagement_lagged)
    print(f"Correlation (polarization -> engagement, {lag_hours}h lag): {corr_lag:.3f}")
    
    # Identify feedback loop periods
    # Both polarization and engagement are high and increasing
    feedback_periods = []
    
    for i in range(1, len(polarization_df)):
        curr = polarization_df.iloc[i]
        prev = polarization_df.iloc[i-1]
        
        # Check if both are increasing and above average
        if (curr['polarization_index'] > prev['polarization_index'] and
            curr['avg_engagement'] > prev['avg_engagement'] and
            curr['polarization_index'] > polarization_df['polarization_index'].mean() and
            curr['avg_engagement'] > polarization_df['avg_engagement'].mean()):
            
            feedback_periods.append(polarization_df.index[i])
    
    print(f"Identified {len(feedback_periods)} potential feedback loop periods")
    
    return feedback_periods, corr_0, corr_lag

def visualize_temporal_analysis(polarization_df, events, feedback_periods):
    """
    Create temporal visualizations for polarization trends and events.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # 1. Polarization index over time
    axes[0].plot(polarization_df.index, polarization_df['polarization_index'], 
                color='red', linewidth=2, label='Polarization Index')
    axes[0].axhline(y=0.7, color='orange', linestyle='--', alpha=0.7, label='Escalation threshold')
    axes[0].axhline(y=0.5, color='yellow', linestyle='--', alpha=0.7, label='Moderate threshold')
    
    # Mark escalation events
    for event in events:
        axes[0].axvspan(event['start'], event['end'], alpha=0.2, color='red')
    
    # Mark feedback loop periods
    for fp in feedback_periods:
        axes[0].axvline(x=fp, color='purple', alpha=0.3, linestyle=':')
    
    axes[0].set_title('Polarization Index Over Time (French Election 2022)')
    axes[0].set_ylabel('Polarization Index')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Narrative composition over time
    narrative_cols = ['radicalizing_ratio', 'polarizing_ratio', 'dialogue_ratio']
    colors = ['darkred', 'orange', 'green']
    labels = ['Radicalizing', 'Polarizing', 'Dialogue-oriented']
    
    for col, color, label in zip(narrative_cols, colors, labels):
        axes[1].plot(polarization_df.index, polarization_df[col], 
                    color=color, label=label, alpha=0.8)
    
    axes[1].set_title('Narrative Composition Over Time')
    axes[1].set_ylabel('Proportion of Posts')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Engagement vs polarization scatter over time
    scatter = axes[2].scatter(polarization_df['polarization_index'], 
                             polarization_df['avg_engagement'],
                             c=range(len(polarization_df)), 
                             cmap='viridis', alpha=0.6, s=50)
    axes[2].set_xlabel('Polarization Index')
    axes[2].set_ylabel('Average Engagement')
    axes[2].set_title('Engagement vs Polarization (Color = Time progression)')
    plt.colorbar(scatter, ax=axes[2], label='Time')
    
    plt.tight_layout()
    plt.savefig('temporal_analysis_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nVisualization saved as 'temporal_analysis_results.png'")

if __name__ == "__main__":
    # Load processed data
    print("Loading processed data...")
    df = pd.read_csv('france_election_2022_processed.csv')
    
    # Calculate polarization index
    polarization_df = calculate_polarization_index(df, window='6H')
    
    # Detect escalation events
    events = detect_escalation_events(polarization_df, threshold=0.7, min_duration=3)
    
    # Identify feedback loops
    feedback_periods, corr_0, corr_lag = identify_feedback_loops(polarization_df)
    
    # Visualize
    visualize_temporal_analysis(polarization_df, events, feedback_periods)
    
    print("\n=== TEMPORAL ANALYSIS SUMMARY ===")
    print(f"Total time windows analyzed: {len(polarization_df)}")
    print(f"Escalation events detected: {len(events)}")
    print(f"Feedback loop periods: {len(feedback_periods)}")
    print(f"Polarization-engagement correlation: {corr_0:.3f}")
