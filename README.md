# AMATI-FAQ

This script takes a set of student questions from the AMATI project and runs a cluster analysis to generate a FAQ document.

## Setup

Create a venv and install the required libraries

```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Json Data to be processed should be located in the `data` folder. The .json file should contain an **array** of elements with at least the following structure:

```json
{
  "content": "How does this even work?",
  "slideSet": 12
}
```
## Running the script

The script takes to inputs: 

| Option | Short | Description |
| ------ | ----- | ----------- |
| --filename "data/yourdata.json" | -f | Input data to cluster |
| --clusters 5 | -c | Number of cluster for the final plot |
| --no-plot | -np | Disables showing the plot windows. Still saved the final plot to file |

## Outputs

The script first generates the TF-IDF matrix for the input data and then generates a distortion plot for 5 to 15 clusters to choose optimal number of cluster using the [elbow method heuristic](https://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set).
This plot has to be manually saved if you want to keep it!

Then, plots for the [silhouette analysis](https://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set#The_silhouette_method) are generated for 5 to 15 clusters.
Exact values for the average cluster silhouette score are printed to the output.

Lastly a detailed plot for the given number of clusters is saved in clusters.png and the top identifying terms for each cluster are printed to output.
