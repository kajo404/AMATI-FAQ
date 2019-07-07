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

## Cluster analysis

The `cluster.py` takes these inputs: 

| Option | Short | Description |
| ------ | ----- | ----------- |
| --filename "data/yourdata.json" | -f | Input data to cluster |
| --clusters 5 | -c | Number of cluster for the final plot |
| --no-plot | -np | (Optional) Disables showing the plot windows. Still saved the final plot to file |

Example:

```
python cluster.py -f "data/yourdata.json" -c 5
```

### Outputs

The script first generates the TF-IDF matrix for the input data and then generates a distortion plot for 5 to 15 clusters to choose optimal number of cluster using the [elbow method heuristic](https://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set).
This plot has to be manually saved if you want to keep it!

Then, plots for the [silhouette analysis](https://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set#The_silhouette_method) are generated for 5 to 15 clusters.
Exact values for the average cluster silhouette score are printed to the output.

Lastly a detailed plot for the given number of clusters is saved in clusters.png and the top identifying terms for each cluster are printed to output.

## FAQ Prediction

The `predict.py` script takes these inputs:

| Option | Short | Description |
| ------ | ----- | ----------- |
| --input | -i | Input of data to predict against FAQ model |
| --faq | -faq | FAQ data in .json format to build the tf-idf model |

The input needs to be a `json` array where each element has a `content` field

The FAQ data should also be an array consisting of elements with `question` and `answer` fields



