import matplotlib.pyplot as plt
import squidpy as sq
import scanpy as sc

def plot_clusters(ad, typical_scales, resolution):
    scales_plot = ['Raw'] + ['scale' + str(scale) for scale in typical_scales]
    
    fig, axes = plt.subplots(
        len(typical_scales)+1, 2,
        figsize=(9, len(typical_scales)*3),
        dpi=100)
    for i, scale in enumerate(scales_plot):
        cluster_key = 'leiden_' + str(scale) + '_res' + str(resolution)
        ad.obsm['X_umap'] = ad.obsm['X_umap_' + str(scale)]
        sc.pl.umap(
            ad,  
            color=cluster_key, 
            size=1, 
            ax=axes[i, 0], 
            show=False, 
            title='UMAP of ' + str(scale)
        )
        for collection in axes[i, 0].collections:
            collection.set_rasterized(True)
        # hide legend
        axes[i, 0].legend_.remove()

        sq.pl.spatial_scatter(
            ad, 
            library_id="spatial", 
            shape=None, 
            color= cluster_key,
            wspace=0.1,
            frameon=False,
            size=1,
            figsize=(18, 18),
            dpi=100,
            outline=False,
            img=False,
            marker='.',
            ax=axes[i, 1],
            title='Cell clusters\n scale={scale}, resolution={resolution}'.format(scale=scale, resolution=resolution),
        )
        for collection in axes[i, 1].collections:
            collection.set_rasterized(True)
    plt.tight_layout()
    plt.show()
    plt.close()
