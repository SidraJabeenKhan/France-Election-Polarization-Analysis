# France Election Polarization Analysis

Computational social science analysis of electoral polarization on X/Twitter during French national elections, developed for the DEEP DIALOGUE Horizon Europe PhD application at Lund University.

## Project Overview

This repository demonstrates a complete workflow for analyzing how electoral campaigns on social media create feedback loops between polarization and radicalization. The project focuses on French national elections as a case study, with methodological frameworks applicable to other European electoral contexts including Italy, Hungary, and Romania.

The analysis addresses core research questions from the DEEP DIALOGUE project:
- When, how, and why does polarization turn into radicalization among European youth?
- What narrative patterns predict escalation versus de-escalation?
- How can early-warning indicators be designed for democratic institutions?

## Analytical Components

### 1. Text Analysis & Narrative Classification
- Social media text preprocessing and cleaning
- Sentiment analysis using lexicon-based and machine learning approaches
- Topic modeling (LDA) to identify emergent themes
- Custom narrative taxonomy: polarization intensity, radicalization indicators, dialogue orientation

### 2. Network Analysis
- Retweet network construction and visualization
- Community detection using modularity optimization
- Influencer identification via centrality metrics
- Echo chamber detection through clustering coefficients

### 3. Temporal Analysis
- Time series analysis of polarization trends
- Event-based escalation detection (debates, scandals, electoral milestones)
- Before/during/after electoral period comparison
- Feedback loop identification between polarization and radicalization

### 4. Geographic Visualization
- Choropleth mapping of narrative intensity across French regions
- Department-level polarization indices
- Territorial political geography analysis
- Connection between digital polarization and electoral geography

### 5. Early-Warning Dashboard
- Real-time polarization index calculation
- Threshold-based alert system for narrative escalation
- Multi-metric dashboard prototype (polarization, network fragmentation, sentiment volatility)
- Visualization interface for democratic institutions

## Technical Stack

- **Python 3.10+**
- **Data Manipulation**: pandas, numpy
- **Text Analysis**: NLTK, spaCy (French model), TextBlob, VADER
- **Topic Modeling**: gensim (LDA)
- **Network Analysis**: networkx, igraph
- **Visualization**: matplotlib, seaborn, plotly, folium, geopandas
- **Geographic Data**: geopandas, shapefile processing
- **Dashboard**: plotly Dash (prototype)

## Data Sources

This project uses simulated/synthetic data that mirrors the structure and patterns of real social media data from French electoral contexts. The simulation is based on:

- French electoral hashtag patterns (#Presidentielle2022, #Presidentielle2027, #Macron, #LePen, #Melenchon, #RN, #LFI, #Republique, #Democratie)
- Temporal patterns from actual French electoral cycles
- Geographic distribution based on French electoral geography (urban/rural, regional political traditions)
- Network structures reflecting known social media polarization dynamics

**Note**: For actual research, data would be collected via X/Twitter API (tweepy) in compliance with platform terms of service and GDPR requirements.

## Skills Demonstrated

This repository showcases the following computational social science competencies:

- **Social Media Data Collection**: API interaction concepts, sampling strategies, hashtag-based and keyword-based collection
- **Text Mining**: Preprocessing, tokenization, stopword removal, lemmatization, sentiment analysis, named entity recognition
- **Machine Learning for Text**: Topic modeling (LDA), narrative classification, feature extraction
- **Network Science**: Graph construction, community detection, centrality analysis, temporal networks
- **Time Series Analysis**: Trend detection, event identification, seasonal decomposition, forecasting concepts
- **Geospatial Analysis**: Coordinate mapping, choropleth visualization, regional aggregation, spatial autocorrelation
- **Dashboard Development**: Multi-metric visualization, real-time updating concepts, user interface design for non-technical stakeholders
- **Research Ethics**: Data privacy considerations, anonymization, platform terms compliance, vulnerable population protection

## Certification

Python for Data Science — Great Learning (June 2026)
R Programming for Data Science — Great Learning (June 2026)

## Author

Sidra Jabeen Khan
PhD Researcher | Computational Social Science | Political Communication | Gender History & Postcolonial Studies

- GitHub: https://github.com/SidraJabeenKhan
- ResearchGate: https://www.researchgate.net/profile/Sidra-Jabeen-Khan
- LinkedIn: https://www.linkedin.com/in/sidra-jabeen-khan-25198344

## Project Context

This repository was developed as part of a PhD application for the DEEP DIALOGUE project at Lund University, funded by Horizon Europe. The project examines when, how, and why polarization turns into radicalization among European youth, maps feedback loops between polarization and radicalization, and tests dialogue-based interventions that promote democratic participation, trust, and social cohesion.

## License

MIT License — open for academic and research use.
