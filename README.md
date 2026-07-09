# Tourist-attraction-recommendations

# Hybrid Tourism Recommender System

This repository contains the architecture, dataset, and inference logic for a geospatial-aware tourism recommendation system. The system addresses the intrinsic limitations of traditional Collaborative Filtering (CF) and lexical-based Content-Based Filtering (CBF) by introducing a tripartite fusion architecture.

## 🧠 System Architecture

The recommendation engine is built upon three core pillars:
1. **Semantic NLP (IndoSBERT):** Replaces rigid TF-IDF with `paraphrase-multilingual-MiniLM-L12-v2` to map user queries into a high-dimensional dense vector space, enabling bidirectional semantic context understanding and resolving lexical mismatches.
2. **Search Space Partitioning (K-Means):** Partitions the geospatial tourism data into 3 distinct clusters (K=3). This ensures processing efficiency and mitigates the sparse-data elimination risk commonly found in density-based algorithms like DBSCAN.
3. **Geospatial Filtering (Haversine Formula):** Acts as an absolute spatial constraint, deterministically trimming geographically infeasible destinations from the recommendation pool.

**Fusion Weighting:** The tripartite ranking normalizes and fuses Semantic (40%), CF (40%), and Reputation (20%) scores, consistently achieving a Hit Rate @ 5 above 82%.

## 📂 Repository Structure

*   `Tourism_Data.csv`: The raw dataset (85 MB) scraped from Google Maps, containing 5,717 entities. Features include place name, coordinates, user reviews, ratings, and facility metadata.
*   `Tourist_Attraction.ipynb`: The primary Jupyter Notebook containing the data pipeline, preprocessing (RobustScaler, IQR), K-Means clustering execution, IndoSBERT encoding, and the final hybrid inference logic.

## 🚀 Execution Guide

To replicate the results or run the inference system:
1. Clone this repository locally.
2. Ensure you have a Python environment with necessary dependencies (`pandas`, `scikit-learn`, `sentence-transformers`, `numpy`).
3. Open `Tourist_Attraction.ipynb` in Jupyter or Google Colab.
4. Run the cells sequentially. The notebook will automatically ingest `Tourism_Data.csv` from the local directory.
