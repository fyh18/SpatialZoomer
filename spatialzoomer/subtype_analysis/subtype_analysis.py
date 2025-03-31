from .CFSFDP import CFSFDP
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib as mpl  
import seaborn as sns
from kneed import KneeLocator
import os

def generate_colors(num_colors):
    cmap_list = [plt.cm.tab20, plt.cm.tab20b, plt.cm.tab20c, plt.cm.tab10, plt.cm.Paired, plt.cm.Set3]
    colors = np.vstack([cmap(np.linspace(0, 1, cmap.N))[:, :3] for cmap in cmap_list])
    if len(colors) < num_colors:
        additional_colors = np.random.rand(num_colors - len(colors), 3)
        colors = np.vstack([colors, additional_colors])
    return colors[:num_colors]

# global variables
colors_3 = ['#CD69C9', '#9ACD32', "#FF7F00"]
new_colors = generate_colors(200)
config = {
    "font.family":'sans-serif',
    "font.size": 12,
    "mathtext.fontset":'stix',
    "font.serif": ['MS Arial'],
}
rcParams.update(config)
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


def calculate_centroids(X, labels):
    unique_labels = np.unique(labels)
    centroids = np.vstack([X[labels == label].mean(axis=0) for label in unique_labels])
    return centroids, unique_labels

def ward_linkage(X, y):
    centroids, unique_labels = calculate_centroids(X, y)
    sizes = np.array([np.sum(y == label) for label in unique_labels])

    all_centroids_exp = centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    centroid_dists_sq = np.sum(all_centroids_exp ** 2, axis=-1)
    ess_diff = (sizes[:, np.newaxis] * sizes[np.newaxis, :]) / (sizes[:, np.newaxis] + sizes[np.newaxis, :]) * centroid_dists_sq
    
    # 确保只考虑上三角部分的有效合并差值
    mask = np.triu(np.ones_like(ess_diff, dtype=bool), k=1)
    ess_diff[~mask] = np.inf

    min_ward_value = np.min(ess_diff)
    return min_ward_value# / len(y)


def calculate_metrics(ad, save_path, nonepi=True, cluster_key='leiden_Raw_res1'):
    """
    Plot covariance-scale curves and covariance values.
    input ad: AnnData object, with umaps at raw ("X_umap_Raw") and different scales (for example "X_umap_scale0.01")
    input cluster_key: key of the clustering result in ad.obs
    input save_path: path to save the figures
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ######################### plot umap #########################
    # set as categorical
    ad.obs[cluster_key] = ad.obs[cluster_key].astype('category')
    cell_type_counts = ad.obs[cluster_key].value_counts()
    sorted_cell_types = cell_type_counts.index.tolist()
    color_mapping = {cell_type: new_colors[i] for i, cell_type in enumerate(sorted_cell_types)}
    sorted_colors = [color_mapping[cell_type] for cell_type in ad.obs[cluster_key].cat.categories]
    ad.uns[cluster_key + '_colors'] = sorted_colors
    ad.obsm['X_umap'] = ad.obsm['X_umap_Raw']

    fig, ax = plt.subplots(figsize=(5, 5))
    sc.pl.umap(ad, 
            color=cluster_key, 
            size=5, 
            ax=ax, 
            show=False, 
            title='Cell Types')
    for collection in ax.collections:
        collection.set_rasterized(True) 
    plt.savefig(save_path + "/umap_clusters.pdf", bbox_inches='tight', dpi=300)
    
    ######################### plot covariance-scale curves #########################
    obsm_key = list(ad.obsm.keys())
    umap_keys = [key for key in obsm_key if key.startswith('X_umap_scale')]
    umap_keys = sorted(umap_keys, key=lambda x: float(x.split('scale')[-1]))
    umap_keys = [umap_key for umap_key in umap_keys if float(umap_key.split('scale')[-1]) <= 10.0]
    
    # normalize umap
    for umap_scale in umap_keys:
        # X.obsm['umap_scale']归一化
        ad.obsm['scaled'+umap_scale] = (ad.obsm[umap_scale] - ad.obsm[umap_scale].min()) / (ad.obsm[umap_scale].max() - ad.obsm[umap_scale].min())
        ad.obsm['scaled_X_umap_Raw'] = (ad.obsm['X_umap_Raw'] - ad.obsm['X_umap_Raw'].min()) / (ad.obsm['X_umap_Raw'].max() - ad.obsm['X_umap_Raw'].min())
    
    # loop over clusters and scales to calculate variance
    nmf_clusters = ad.obs[cluster_key].unique()
    var_dict = {}
    k_dict = {}
    for nmf_cluster in nmf_clusters:
        var_list = []
        for umap_scale in umap_keys:
            X = ad[ad.obs[cluster_key] == nmf_cluster].obsm['scaled'+umap_scale]
            var = np.var(X, axis=0).mean()
            var_list.append(var)
        k = np.polyfit([float(x.split('scale')[-1]) for x in umap_keys], var_list, 1)[0]
        var_dict[nmf_cluster] = var_list
        k_dict[nmf_cluster] = k

    if nonepi:
        var_dict = {k: v for k, v in var_dict.items() if "Tumor Cells" not in k 
                        and "Epithelial" not in k 
                        and "Epithelium" not in k
                        and "AT2" not in k 
                        and "Unassigned" not in k 
                        and "Ciliated" not in k
                        and "Malignant" not in k}
        k_dict = {k: v for k, v in k_dict.items() if "Tumor Cells" not in k 
                        and "Epithelial" not in k 
                        and "Epithelium" not in k
                        and "AT2" not in k 
                        and "Unassigned" not in k 
                        and "Ciliated" not in k
                        and "Malignant" not in k}

    k_dict = dict(sorted(k_dict.items(), key=lambda x: x[1], reverse=False))
    clusters = list(k_dict.keys())
    values = list(k_dict.values())


    fig, axs = plt.subplots(1, 2, figsize=(8, 3))  # 调整画布大小以适应布局

    # plot variance-scale curves for each cluster on the left
    for nmf_cluster, var_list in var_dict.items():
        axs[0].plot(var_list, label=nmf_cluster, color=color_mapping[nmf_cluster])
    # 移除图例和数值
    axs[0].set_xticks(range(len(umap_keys)))
    axs[0].set_xticklabels([x.split('scale')[-1] for x in umap_keys])  # x轴标签

    # bar plot of k values for each cluster on the right
    right_ax = axs[1]
    right_ax.barh(clusters, values, color=[color_mapping[cluster] for cluster in clusters])

    right_ax.set_yticks(range(len(clusters)))
    right_ax.set_yticklabels(clusters)
    right_ax.set_title("k values")


    plt.tight_layout()
    plt.savefig(save_path + "/var_plot_withk.pdf", dpi=300)

    return umap_keys, var_dict, k_dict, clusters[-1] # return var_dict, k_dict, knee_dict, and the cluster with the largest k value

def kde_plot(ad_selected, save_path):
    """
    Plot KDE plots for each cluster.
    input ad: AnnData object, with umaps at raw ("X_umap_Raw") and different scales (for example "X_umap_scale0.01")
    input cluster_key: key of the clustering result in ad.obs
    input save_path: path to save the figures
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    obsm_key = list(ad_selected.obsm.keys())
    umap_keys = [key for key in obsm_key if key.startswith('X_umap_scale')]
    umap_keys = sorted(umap_keys, key=lambda x: float(x.split('scale')[-1]))
    umap_keys = [umap_key for umap_key in umap_keys if float(umap_key.split('scale')[-1]) <= 10.0]

    fig, axes = plt.subplots(3, int((len(umap_keys)+2)/3), figsize=(len(umap_keys), 6))

    # # raw
    # X = ad_selected.obsm['X_umap_Raw']
    # ax = axes.flatten()[0]

    # # KDE plot
    # ax.scatter(X[:, 0], X[:, 1], color='gray', s=0.5)
    # for collection in ax.collections:
    #     collection.set_rasterized(True)  # 将散点设置为栅格化

    # umap_df = pd.DataFrame(X, columns=['UMAP1', 'UMAP2'])
    # sns.kdeplot(data=umap_df, x='UMAP1', y='UMAP2', ax=ax, bw_adjust=2,
    #             cmap='Blues', fill=True, thresh=0.01, levels=8, alpha=0.8)
    # ax.set_title('X_umap_Raw')

    for key in umap_keys:
        X = ad_selected.obsm[key]
        ax = axes.flatten()[umap_keys.index(key)]
        ax.scatter(X[:, 0], X[:, 1], color='gray', s=0.05, alpha=0.5)
        for collection in ax.collections:
            collection.set_rasterized(True)  # 将散点设置为栅格化
        # KDE plot
        umap_df = pd.DataFrame(X, columns=['UMAP1', 'UMAP2'])
        sns.kdeplot(data=umap_df, x='UMAP1', y='UMAP2', ax=ax, bw_adjust=2,
                    cmap='Blues', fill=True, thresh=0.01, levels=8, alpha=0.7)
        ax.set_title(key)
    plt.tight_layout()
    plt.savefig(save_path + "/umaps_kde.pdf", dpi=300)

def find_optimal_scale(var_list, umap_keys):
    kl = KneeLocator(
        range(len(var_list)), 
        var_list, 
        curve='concave', 
        direction='increasing', 
        online=True,
        S=0.1,
    )
    optimal_scale = umap_keys[kl.elbow]
    optimal_scale = optimal_scale.split('scale')[-1]
    return optimal_scale

def detect_subclusters(ad_selected, save_path, optimal_scale, n_clusters=2, plot_all_scales=True, calculate_ward=True):
    ad_selected.obsm['X_umap'] = ad_selected.obsm['X_umap_scale' + str(optimal_scale)]
    clustering_model = CFSFDP(n_clusters=n_clusters)
    res = clustering_model.fit_predict(ad_selected.obsm['X_umap'])
    ad_selected.obs['cluster_label'] = list(res.astype(str))

    ad_selected_plot = ad_selected[ad_selected.obs['cluster_label'] != '-1']
    ad_selected_plot.uns['cluster_label_colors'] = [colors_3[1], colors_3[0]]
    fig, ax = plt.subplots(figsize=(5, 5))
    sc.pl.umap(ad_selected_plot, color=['cluster_label'], title='subclusters at scale' + str(optimal_scale), 
            show=False, ax=ax)
    plt.savefig(save_path + "/umap_subclusters_scale" + str(optimal_scale) + ".pdf", dpi=300)

    obsm_key = list(ad_selected.obsm.keys())
    umap_keys = [key for key in obsm_key if key.startswith('X_umap_scale')]
    umap_keys = sorted(umap_keys, key=lambda x: float(x.split('scale')[-1]))
    umap_keys = [umap_key for umap_key in umap_keys if float(umap_key.split('scale')[-1]) <= 10.0]

    if calculate_ward:
        ward_list = dict()
        for key in umap_keys:
            X = ad_selected_plot.obsm['scaled'+key]
            y = ad_selected_plot.obs['cluster_label']
            ward_list[key] = ward_linkage(X, y)

    if plot_all_scales:
        fig, axes = plt.subplots(3, int((len(umap_keys)+2)/3), figsize=(len(umap_keys)*2, 12))

        # # raw
        # ad_selected_plot.obsm['X_umap'] = ad_selected_plot.obsm['X_umap_Raw']
        # ax = axes.flatten()[0]
        # sc.pl.umap(ad_selected_plot, 
        #             size=20,
        #             color=['cluster_label'], title='X_umap_Raw', 
        #             ax=ax, 
        #             show=False)
        # for collection in ax.collections:
        #     collection.set_rasterized(True)

        for key in umap_keys:
            ad_selected_plot.obsm['X_umap'] = ad_selected_plot.obsm[key]
            ax = axes.flatten()[umap_keys.index(key)]
            sc.pl.umap(ad_selected_plot, 
                    size=15,
                    color=['cluster_label'], title=key, 
                    ax=ax, 
                    show=False)
            for collection in ax.collections:
                collection.set_rasterized(True)
            if calculate_ward:
                ax.set_title(key + '\n ward: %.2f' % ward_list[key])
            else:
                ax.set_title(key)
        
        plt.tight_layout()
        plt.savefig(save_path + "/umaps_cluster_scale" + str(optimal_scale) + ".pdf", dpi=300)

    return ad_selected_plot, ward_list    # return ad_selected_plot with subcluster labels (exclude -1) and ward_list



def deg2df(deg_groups, cluster_name):
    result_df = pd.DataFrame({key: deg_groups[key][cluster_name] for key in ['names', 'scores', 'pvals', 'pvals_adj', 'logfoldchanges']})
    return result_df

def deg_analysis(ad_selected_plot, save_path, filter_background=True, calculate_ratio=False):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    

