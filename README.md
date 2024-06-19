# Dynamic Time Warping and Clustering for Precipitation Data

## Overview

This project, conducted at the Altmos Lab of the University of Versailles, involves collaboration with a team of researchers, including statisticians and physicists, to analyze the behavior of rain over a period of 20 years. The data consists of precipitation measurements at 5-minute intervals, covering a 300x300 km grid around Paris. Given the vast amount of data and limited computational resources, innovative methods were developed to handle and analyze these large datasets efficiently.

### Methods

1. **Dynamic Time Warping (DTW) for Distance Measurement:**
   - DTW is a powerful technique for measuring similarity between two time series. Unlike traditional Euclidean distance, DTW can handle shifts and distortions in the time axis, making it ideal for comparing precipitation patterns that may vary in timing and intensity.

2. **Sliding Window Approach for Local Clustering:**
   - To manage the computational load, the data was divided into smaller, overlapping sub-sections using a sliding window approach. This method allows for localized clustering of precipitation patterns, reducing the memory and computational requirements.

3. **Precomputed Distance Matrices:**
   - Distance matrices were precomputed for each sub-section and stored. This significantly speeds up the clustering process, as the expensive computation of DTW distances is done only once and reused multiple times.

4. **Parallel Processing:**
   - Leveraging parallel processing capabilities, the calculations of distance matrices and clustering operations were distributed across multiple processors. This parallelism reduces the overall processing time and makes it feasible to handle the large datasets.

5. **Adaptive DBSCAN Clustering:**
   - DBSCAN (Density-Based Spatial Clustering of Applications with Noise) was employed to identify clusters within each sub-section. This algorithm is well-suited for the irregular and non-linear nature of precipitation data. Parameters like epsilon (eps) and minimum samples were tuned to adapt to different regions and time periods.

6. **Merging Clusters with Shared Core Points:**
   - To ensure consistency and continuity across overlapping sub-sections, clusters sharing core points were merged. This method helps in maintaining the integrity of precipitation patterns across the entire grid.

7. **Visualization Techniques:**
   - Enhanced visualization methods were developed to display clustering results, including highlighting core points and generating comprehensive cluster maps. These visual tools aid in the interpretation and presentation of the findings.




1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/your-repository.git
   cd your-repository
![image](https://github.com/chahineNejm/precipitation-LATMOS/assets/150661156/ab8ce608-8c1e-4d94-875c-9e1034666889)

The image shows the results of the DBSCAN clustering performed on the time series data for June 2018. The image reveals an increase in turbulence activity above the Paris metropolitan area, suggesting a possible influence of this urban zone on precipitation. This result is consistent with several articles we have consulted, which state that the Parisian heat island tends to destabilize the local air mass in summer, leading to an increase in the number of thunderstorms.

