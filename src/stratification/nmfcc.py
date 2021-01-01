# TEST

import numpy as np
import pandas as pd
from collections import defaultdict

import nimfa
from sklearn import preprocessing
import matplotlib.pyplot as plt

# for heatmap and hierarchical clustering
import matplotlib.gridspec as gridspec
import scipy.cluster.hierarchy as sch
from scipy.spatial import distance
from scipy.cluster import hierarchy
import random
import seaborn as sns

from data.reddy_data.reddy_dataset import ReddyDataset

# create distinct colours for clustermap visualisation
distinct_colours = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128), (0, 0, 0), (0, 0, 0),(0, 0, 0),(0, 0, 0),(0, 0, 0),(0, 0, 0),(0, 0, 0),(0, 0, 0),(0, 0, 0),(0, 0, 0),(0, 0, 0)]
distinct_colours = [(r/255, g/255, b/255) for (r,g,b) in distinct_colours]

class NMFCCRun():
    """Class to record NMFCC parameters and intermediate data used during an NMFCC stratification run. Also has functions for visualisation.
    TODO: need to have flags for initial reddy data to know what's been changed. give attributes: initial_reddy_dataset, filtered_reddy_dataset (i.e. filtered expression), DF with subgroups after clustering.
    TODO: messy with self.[attribute] setting here. but not really sure how to change.
    TODO: passed in ReddyDataset for visualisation and clinical data subgroup creation.
    TODO: testing: make sure no zeroes in ranks or n_runs on input.
    TODO: sort out passing alpha, beta, n_w, n_h for multiple ranks.
    """
    def __init__(self, filtered_reddy_dataset:ReddyDataset, subset_required:tuple, ranks:list, nmf_method:str, method_hps:dict):
        # following are populated on initialisation
        self.filtered_reddy_dataset = filtered_reddy_dataset       # pass this own so can     
        self.input_data = filtered_reddy_dataset.expression_data  # input data to algorithm as pandas DF
        self.subset_required = subset_required  # tuple of ints (num_patients, num_genes)
        self.ranks = ranks                      # list of input ranks
        self.nmf_method = nmf_method            # nmf method to be used
        self.method_hps = method_hps            # hyperparameters for specific nmf method
        
        # functions populate following attributes
        self.patients = None    # pd.Int64Index; patients in order of chosen subset
        self.summary = None     # dict; summary of chosen nmf method at different ranks
        self.best_rank = None   # int; best rank determined by metric score
        self.consensus = None   # np.matrix; consensus matrix at best rank
        self.d = None
        self.index_with_cluster = None # list; kept for visualisation
        self.subgroups = None   # dict; {subgroup_id: [patients_ids]}
        self.output_data = None # pd.DataFrame; input data with subgroup assignments

    # input -> numpy subset -> run nmf w hps -> consensus -> linkage -> output w subgroups 
    def create_subgroups_from_input(self): 
        to_cluster = self.create_subset_for_clustering(self.input_data, self.subset_required)
        # initial checks -> should maybe add these to __init__()
        if type(self.ranks) is not list:
            raise TypeError("Ranks needs to be inputted as a list.")
        if self.ranks == [0]:
            return ValueError("Rank must be passed as integer list with integers only from 1 to 20.")
        n_runs = self.method_hps['n_runs'] if (self.method_hps['n_runs'] is not None) else 5  # default n_runs = 5
        
        if self.nmf_method == 'NMF':
            summary = self.create_nmf_summary(to_cluster, self.ranks, n_runs)
        elif self.nmf_method == 'PNMF': 
            summary = self.create_pnmf_summary(to_cluster, self.ranks, n_runs)
        elif self.nmf_method == 'ICM' or self.nmf_method == 'BD': # for both of these, need to sort hps that depend on rank on multiple iterations
            alpha_bnmf = self.method_hps['alpha_bnmf'] if (self.method_hps['alpha_bnmf'] is not None) else np.random.randn(to_cluster.shape[0], self.ranks[0]) # basis W prior
            beta_bnmf = self.method_hps['beta_bnmf'] if (self.method_hps['beta_bnmf'] is not None) else np.random.randn(self.ranks[0], to_cluster.shape[1]) # mixture H prior
            theta_bnmf = self.method_hps['theta_bnmf'] if (self.method_hps['theta_bnmf'] is not None) else 0    # theta
            k_bnmf = self.method_hps['k_bnmf'] if (self.method_hps['k_bnmf'] is not None) else 0                # k
            sigma_bnmf = self.method_hps['sigma_bnmf'] if (self.method_hps['sigma_bnmf'] is not None) else 1    # sigma
            if self.nmf_method == 'ICM':
                summary = self.create_icm_summary(to_cluster, self.ranks, n_runs, alpha_bnmf, beta_bnmf, theta_bnmf, k_bnmf, sigma_bnmf)
            elif self.nmf_method == 'BD':
                n_w = self.method_hps['n_w'] if (self.method_hps['n_w'] is not None) else np.zeros((self.ranks[0], 1)) # n_w default: all false
                n_h = self.method_hps['n_h'] if (self.method_hps['n_h'] is not None) else np.zeros((self.ranks[0], 1)) # n_h default: all false
                summary = self.create_bd_summary(to_cluster, self.ranks, n_runs, alpha_bnmf, beta_bnmf, theta_bnmf, k_bnmf, sigma_bnmf, n_w, n_h)
        else:
            raise ValueError("The provided label does not correspond to an available NMFCC algorithm. Possible labels: NMF, PNMF, BD, ICM.")
        best_rank = self.perform_intrinsic_evaluation(summary, self.ranks)
        consensus = self.get_consensus_from_summary(summary)
        linkage = self.get_linkage_from_consensus(consensus)
        subgroups = self.regain_subgroups_from_linkage(linkage)
        # then create output dataframe?
        return subgroups

    def create_subset_for_clustering(self, input_data, subset_required):
        # create subset and list of patient ids
        num_patients, num_genes = subset_required
        subset = input_data.iloc[0:num_patients, 0:num_genes].copy()
        self.patients = subset.index
        # create numpy version of subset for clustering
        to_cluster = subset.to_numpy()
        to_cluster = preprocessing.Normalizer().fit_transform(to_cluster)
        to_cluster = to_cluster.T   # .T since clustering on patients
        return to_cluster

    def create_nmf_summary(self, data, ranks, n_runs):
        if any(arg is None for arg in [ranks, n_runs]):
            raise ValueError("Either ranks or n_runs is empty. Recreate the NMFCC class with these parameters inputted.")
        nmf = nimfa.Nmf(data, seed="random_vcol")
        summary = nmf.estimate_rank(rank_range=ranks, n_run=n_runs, what='all')
        self.summary = summary
        return summary

    def create_pnmf_summary(self, data, ranks, n_runs):
        pmf = nimfa.Pmf(data, seed="random_vcol", max_iter=80, rel_error=1e-5)
        summary = pmf.estimate_rank(rank_range=ranks, n_run=n_runs, what='all')
        self.summary = summary
        return summary

    # bayesian nmf methods: icm, bd
    # issue: alpha, beta, n_w, n_h depend on rank and so change each iteration when more than one rank -> i'm not sure if nimfa does this already
    # could use track factor method and manually create my own summary dict [refer to nimfa/models/nmf.py estimate_rank()]
    # bd = nimfa.Bd(data, rank=rank, seed="random_c", track_factor=True, n_run=n_runs, alpha=alpha_bnmf, ...)
    # bd_fit = bd() -> get data for each rank and populate summary dict
    # will also need to add option for changing random seeds
    def create_icm_summary(self, data, ranks, n_runs, alpha_bnmf, beta_bnmf, theta_bnmf, k_bnmf, sigma_bnmf):
        if len(ranks) == 1:
            icm = nimfa.Icm(data, seed="random_vcol", alpha=alpha_bnmf, beta=beta_bnmf, theta=theta_bnmf, k=k_bnmf, sigma=sigma_bnmf)
            summary = icm.estimate_rank(rank_range=ranks, n_run=n_runs, what='all')
        elif len(ranks) > 1:
            # Following gives: ValueError: could not broadcast input array from shape (50,3) into shape (50)
            # alphas = [np.random.randn(self.subset_required[1], rank) for rank in ranks]
            # betas = [np.random.randn(rank, self.subset_required[0]) for rank in ranks]
            # icm = nimfa.Icm(data, seed="random_vcol", alpha=alphas, beta=betas, theta=theta_bnmf, k=k_bnmf, sigma=sigma_bnmf)
            icm = nimfa.Icm(data, seed="random_vcol", theta=theta_bnmf, k=k_bnmf, sigma=sigma_bnmf)
            summary = icm.estimate_rank(rank_range=ranks, n_run=n_runs, what='all')
        self.summary = summary
        return summary  

    def create_bd_summary(self, data, ranks, n_runs, alpha_bnmf, beta_bnmf, theta_bnmf, k_bnmf, sigma_bnmf, n_w, n_h):
        if len(ranks) == 1:
            bd = nimfa.Bd(data, seed="random_vcol", alpha=alpha_bnmf, beta=beta_bnmf, theta=theta_bnmf, k=k_bnmf, sigma=sigma_bnmf, n_w=n_w, n_h=n_h)
            summary = bd.estimate_rank(rank_range=ranks, n_run=n_runs, what='all')
        elif len(ranks) > 1:
            # alpha, beta, n_w, n_h based on rank so need to change per rank
            bd = nimfa.Bd(data, seed="random_vcol", theta=theta_bnmf, k=k_bnmf, sigma=sigma_bnmf)
            summary = bd.estimate_rank(rank_range=ranks, n_run=n_runs, what='all')
        self.summary = summary
        return summary

    def get_consensus_from_summary(self, summary):
        if summary is None:
            raise ValueError("Need to run create_nmfcc_summary() first.")    
        consensus = summary[self.best_rank]['consensus']
        self.consensus = consensus
        return consensus

    def get_linkage_from_consensus(self, consensus):
        # consensus -> d -> linkage
        if consensus is None:
            raise ValueError("Consensus matrix has not been defined yet.")
        # set d from consensus
        d = 1 - consensus
        d = pd.DataFrame(d)
        d.set_index(self.patients, inplace=True) # add patient annotation to columns
        d.columns = self.patients
        self.d = d
        # set linkage from d
        dist = distance.squareform(d) # squareform does element-wise in linkage, not row-wise
        linkage = sch.linkage(dist, method='average') # EA: method='ward', metric='cosine'
        self.linkage = linkage
        return linkage

    def regain_subgroups_from_linkage(self, linkage):
        # regain cluster assigments in patient's original ordering
        fl = sch.fcluster(linkage, self.best_rank, criterion='maxclust') # maxclust sets middle param to #clusters = rank
        # zip reordered indices with cluster assignment
        index_with_cluster = list(zip(range(len(self.patients)+1), fl))
        patient_to_cluster = {}
        for (index, cluster_assignment) in index_with_cluster:
            patient_id = self.patients[index]
            patient_to_cluster[patient_id] = cluster_assignment
        self.index_with_cluster = index_with_cluster
        # add patients to their corresponding subgroups for eval
        subgroups = defaultdict(list)
        for (index, cluster_assignment) in index_with_cluster:
            subgroups[cluster_assignment].append(self.patients[index])
        self.subgroups = subgroups
        return subgroups
    
    def perform_intrinsic_evaluation(self, summary, ranks):
        """Plot metrics at different ranks then pick best rank based on combined metric. Allows for comparison with stratification methods that use hierarchical clustering. 
        Pass in self.summary and self.ranks.
        """
        if any(arg is None for arg in [summary, ranks]):
            raise ValueError("Either summary or ranks is empty. Recreate the NMFCCRun with ranks inputted.")
        
        # need to way to display these plots next to each other
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 7))
        plot_title = self.nmf_method + ": Cophenetic correlation index/dispersion index, RSS, combined metric plots against rank"
        fig.suptitle(plot_title)
        print("Note the different y-axes ranges.")
        cci_vals, disp_vals = self.plot_and_get_cci_and_disp(summary, ranks, ax1)
        rss_vals = self.plot_and_get_rss(summary, ranks, ax2)
        best_rank = self.plot_and_evaluate_cm(cci_vals, disp_vals, rss_vals, ranks)
        self.best_rank = best_rank
        return best_rank

    def plot_and_get_cci_and_disp(self, summary, ranks, ax):
        """Plot cophenetic correlation index and dispersion index against rank on same chart.
        """
        cci_vals = [summary[rank]['cophenetic'] for rank in ranks]
        disp_vals = [summary[rank]['dispersion'] for rank in ranks]
        ax.plot(ranks, cci_vals, 'o-', label='Cophenetic correlation',linewidth=2)
        ax.plot(ranks, disp_vals,'o-', label='Dispersion', linewidth=2)
        ax.legend(loc='lower center',  ncol=3, numpoints=1);
        cd_title = self.nmf_method + ": CCI and DI scores as rank is varied"
        ax.set_title(cd_title)
        ax.set(xlabel='Rank', ylabel='Metric Scores')
        return cci_vals, disp_vals

    def plot_and_get_rss(self, summary, ranks, ax):
        """Plot residual sum of squares against rank.
        """
        rss_vals = [summary[rank]['rss'] for rank in ranks]
        ax.plot(ranks, rss_vals, 'og-', label='RSS', linewidth=2)
        rss_title = self.nmf_method + ": RSS score as rank is varied"
        ax.set_title(rss_title)
        ax.set(xlabel='Rank', ylabel='RSS Score')
        return rss_vals

    def plot_and_evaluate_cm(self, cci_vals, disp_vals, rss_vals, ranks):
        """Plot combined metric against rank and select best-performing rank.
        """
        # scale RSS for CM -> this should be improved
        max_rss = max(rss_vals)
        scaled_rss_vals = [rss/max_rss for rss in rss_vals]
        # scale rank for CM
        max_rank = 20   # should make this a proper constant
        scaled_ranks = [rank/max_rank for rank in ranks]
        # combine metrics in list for each rank
        rank_metrics = list(zip(ranks, cci_vals, disp_vals, scaled_rss_vals, scaled_ranks))
        final_scores = []
        for (rank, cci, disp, scaled_rss, scaled_rank) in rank_metrics:
            cm = 0.5*(cci + disp) - 0.1*(scaled_rss + scaled_rank)
            final_scores.append((rank, cm))
        # plot cm scores
        plt.plot(*zip(*final_scores), 'ob-', linewidth=2)
        cm_title = self.nmf_method + ": Combined metric score as rank is varied"
        plt.title(cm_title)
        plt.ylabel("CM Score")
        plt.xlabel("Rank")
        # pick best rank based on CM score
        # TODO: test for 1 vs list vs null ranks 
        best_rank = max(final_scores, key=lambda x: x[1])[0]
        return best_rank

    ####### START OF VISUALISATION FUNCTIONS

    def visualise(self):
        print("We have performed hierarchical clustering on a consensus matrix and so will plot a clustermap...")
        self.plot_clustermap(self.best_rank, self.subgroups, self.index_with_cluster, self.linkage, self.d, self.nmf_method)

    # need to pass in num_patients, num_genes better -> will have abstract visualise() method in superclass
    def plot_clustermap(self, rank, subgroups, index_with_cluster, linkage, d, nmf_method):
        # generate colours for n subgroups
        colmap = self.generate_colmap_for_subgroups(rank, subgroups) 
        # generate subtype colour information
        subtype_colour_info_list = self.filtered_reddy_dataset.subtypes.subtype_colour_info_list
        # hacky way to get round null but need to sort this
        subtype_colour_info_list = [(patient_cols, colourmap) for (patient_cols, colourmap) in subtype_colour_info_list if 
            (patient_cols is not None) and (colourmap is not None)]
        subtype_colourmaps = [subtype_cm for (subtype_cm, _) in subtype_colour_info_list]
        # get subgroup assignments in order of dendrogram for colouring
        index_to_cluster = dict(index_with_cluster)
        reordered_ind = sch.leaves_list(linkage)
        # plot clustermap
        #g = sns.clustermap(d, metric='euclidean', row_linkage=linkage, col_linkage=linkage)
        g = sns.clustermap(d, metric='euclidean', row_linkage=linkage, col_linkage=linkage, col_colors=subtype_colourmaps, yticklabels=False, xticklabels=False, dendrogram_ratio=(0.1, 0.1), cbar_pos=(1, .5, .03, .3), cbar_kws={'label': 'Distance between patients in consensus matrix'}, tree_kws={'colors':[colmap[index_to_cluster[ca]] for ca in reordered_ind]})
        # don't show row dendrogram: g.ax_row_dendrogram.set_visible(False)
        # set title and labels
        title = "Clustermap on " + nmf_method + " Consensus Matrix of " + str(self.subset_required[0]) + " Patients (" + str(self.subset_required[1]) + " Genes) \n\n"
        g.fig.suptitle(title, y=1.02)
        g.ax_heatmap.set_xlabel("Patients")
        g.ax_heatmap.set_ylabel("Patients")
        # plot legends
        self.create_all_subtype_legends(g, subtype_colour_info_list)

    def generate_colmap_for_subgroups(self, rank, subgroups):
        """Subgroups needs to be a dictionary like {subgroup_id : [list of patient_ids]}.
        """  
        # generate random list of numbers
        num_cols = len(distinct_colours)
        array = list(range(num_cols))
        random.shuffle(array)
        colmap = {}
        # use subgroups.keys() so can reuse subgroup colours
        for sg_id in subgroups.keys():
            colmap[sg_id] = distinct_colours[sg_id]
        return colmap 
    
    def create_all_subtype_legends(self, dendrogram, subtype_colour_info_list):
        for (st_cm, st_bar) in subtype_colour_info_list:
            dendrogram = self.create_subtype_legend(dendrogram, st_bar, st_cm.name)

    # TODO: generalise for more subtypes
    def create_subtype_legend(self, dendrogram, subtype_bar, subtype_name):
        if subtype_name == 'COO Class':
            for (label, colour) in subtype_bar.items():
                dendrogram.ax_col_dendrogram.bar(5, 0, color=colour, label="{}".format(label))
            dendrogram.ax_col_dendrogram.legend(title='COO Class', loc="lower right", ncol=1, bbox_to_anchor=(1.18, -0.5))
        elif subtype_name == 'GRM Level':
            for (label, colour) in subtype_bar.items():
                dendrogram.ax_row_dendrogram.bar(5, 0, color=colour, label="{}".format(label))
            dendrogram.ax_row_dendrogram.legend(title='GRM Level', loc="lower right", ncol=1, bbox_to_anchor=(1, 1.01))
        return dendrogram

    ##### END OF VISUALISATION FUNCTIONS
            