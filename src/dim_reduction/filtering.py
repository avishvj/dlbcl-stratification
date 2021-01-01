import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# for variance threshold filtering
from sklearn.feature_selection import VarianceThreshold
# for Laplacian score filtering
from skfeature.utility import construct_W
from skfeature.function.similarity_based import lap_score

from data.reddy_data.reddy_dataset import ReddyDataset

# TODO: currently have to do VTF then LSF. allow more flexibility in pipeline ordering.
# TODO: allow entering LSF parameters.
# TODO: add functionality for laplacian graph. 
# TODO: filter norm_expression too. ISSUE: vt_threshold needs to be normalised.

class FilterPipeline():
    """ Perform quick filtering methods to reduce the dimensionality of data. Currently for continuous data only.
        
        # 1. users should plot features against threshold to determine best threshold
        # 2. users should then plot lap_score_filters to find best #patients ISSUE: how to use data from both -> need test function
        # 3. perform filtering with best threshold and best num features
    """
    def __init__(self, input_reddy_dataset: ReddyDataset):
        self.input_reddy_dataset = input_reddy_dataset
        self.test_reddy_dataset = input_reddy_dataset
        self.output_reddy_dataset = input_reddy_dataset
        self.laplacian_graph = None
    
    # TODO: input dict {method: hps}
    def perform_filtering(self, vt_threshold, num_features):
        print("Filtering expression data with given parameters...")
        output_expression = self.input_reddy_dataset.expression_data.copy()
        output_expression_vt = self.variance_threshold_selector(output_expression, vt_threshold)
        output_expression_vt_ls = self.lap_score_filtering(output_expression_vt, num_features)
        # create fresh copy in case changed by testing
        print("Populating output reddy data with filtered data...")
        self.output_reddy_dataset.expression_data = output_expression_vt_ls
        print("Expression data filtered. Use this class' output_reddy_dataset for following clustering.")

    def variance_threshold_selector(self, data, threshold=0.5):
        # data = self.input_reddy_dataset.expression_data.copy()
        selector = VarianceThreshold(threshold)
        selector.fit(data)
        return data[data.columns[selector.get_support(indices=True)]]

    # TODO: add features to change parameters of function
    # final lap score once we know
    # can't use this in plot function since requires num_patients
    def lap_score_filtering(self, vt_data, num_features):
        vt_numpy = vt_data.to_numpy()
        # construct affinity matrix
        kwargs_W = {"metric":"cosine","neighbor_mode":"knn","weight_mode":"cosine","k":40, 't':500} 
        print("We perform Laplacian score filtering using the following parameters: " + str(kwargs_W))
        W = construct_W.construct_W(vt_numpy, **kwargs_W)
        score = lap_score.lap_score(vt_numpy, W=W)
        idx = lap_score.feature_ranking(score) # rank features
        filtered_data = vt_data.iloc[:,idx[0:num_features]].copy()
        print("\nThe data now has " + str(len(filtered_data.T)) + " features after Laplacian score filtering.")
        return filtered_data

    # help determine best threshold
    def plot_features_against_vt_threshold(self, thresholds):
        num_features = []
        data = self.test_reddy_dataset.expression_data.copy() 
        for threshold in thresholds:
            vt_data = self.variance_threshold_selector(data, threshold)
            num_features.append(len(vt_data.T))
        plt.plot(thresholds, num_features, '.b-')
        plt.xlabel("Threshold")
        plt.ylabel("Number of features")
        plt.grid(True)
        # plt.axvline(x=0.70, ymax=0.87, color='r')
        # plt.axvline(x=3, ymax=0.2, color='r')
        plt.show()            

    # help determine best number of features after variance thresholding
    def plot_ls_after_vt_filtering(self, threshold):
        data = self.test_reddy_dataset.expression_data.copy()
        vt_data = self.variance_threshold_selector(data, threshold)
        # perform ls filtering
        vt_numpy = vt_data.to_numpy()
        # construct affinity matrix
        kwargs_W = {"metric":"cosine","neighbor_mode":"knn","weight_mode":"cosine","k":40, 't':500} 
        print("We plot the Laplacian scores of the features using the following affinity matrix parameters: " + str(kwargs_W))
        W = construct_W.construct_W(vt_numpy, **kwargs_W)
        # compute lap score of each remaining features
        score = lap_score.lap_score(vt_numpy, W=W)
        self.plot_lap_scores(score)

    def plot_lap_scores(self, score): 
        # plot graph of features in ascending Laplacian score
        sorted_scores = np.sort(score)
        xs = [i for i in range(len(sorted_scores))]
        plt.scatter(xs, sorted_scores)
        plt.xlabel("Feature Index")
        plt.ylabel("Laplacian Score")
        # plt.axvline(x=1500, ymax=(sorted_scores[1500]-0.32)/0.6, color='r')
        plt.show()
