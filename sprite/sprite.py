import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import anndata as ad
import warnings
import os
import gc


def load_paired_datasets (spatial_counts, spatial_loc, RNAseq_counts, spatial_metadata = None,
                          min_cell_prevalence_spatial = 0.0, min_cell_prevalence_RNAseq = 0.01,
                          min_gene_prevalence_spatial = 0.0, min_gene_prevalence_RNAseq = 0.0):
    '''
    Uses datasets in the format specified by Li et al. (2022)
        See: https://drive.google.com/drive/folders/1pHmE9cg_tMcouV1LFJFtbyBJNp7oQo9J
    
    Parameters
    ----------
        spatial_counts [str] - path to spatial counts file; rows are cells
        spatial_loc [str] - path to spatial locations file; rows are cells
        RNAseq_counts [str] - path to RNAseq counts file; rows are genes
        spatial_metadata [None or str] - if not None, then path to spatial metadata file (will be read into spatial_adata.obs)
        min_cell_prevalence_spatial [float between 0 and 1] - minimum prevalence among cells to include gene in spatial anndata object, default=0
        min_cell_prevalence_RNAseq [float between 0 and 1] - minimum prevalence among cells to include gene in RNAseq anndata object, default=0.01
        min_gene_prevalence_spatial [float between 0 and 1] - minimum prevalence among genes to include cell in spatial anndata object, default=0
        min_gene_prevalence_RNAseq [float between 0 and 1] - minimum prevalence among genes to include cell in RNAseq anndata object, default=0
    
    Returns
    -------
        spatial_adata, RNAseq_adata - AnnData objects with counts and location (if applicable) in metadata
    '''
    # Spatial data loading
    spatial_adata = load_spatial_data (spatial_counts,
                                       spatial_loc,
                                       spatial_metadata = spatial_metadata,
                                       min_cell_prevalence_spatial = min_cell_prevalence_spatial,
                                       min_gene_prevalence_spatial = min_gene_prevalence_spatial)
    
    # RNAseq data loading
    RNAseq_adata = load_rnaseq_data (RNAseq_counts,
                                     min_cell_prevalence_RNAseq = min_cell_prevalence_RNAseq,
                                     min_gene_prevalence_RNAseq = min_gene_prevalence_RNAseq)

    return(spatial_adata, RNAseq_adata)


def load_spatial_data (spatial_counts, spatial_loc, spatial_metadata=None,
                       min_cell_prevalence_spatial = 0.0, min_gene_prevalence_spatial = 0.0):
    '''
    Loads in spatial data from text files.
    
    See load_paired_datasets() for details on arguments
    '''
    # read in spatial counts
    df = pd.read_csv(spatial_counts,header=0,sep="\t")
    
    # filter lowly expressed genes
    cells_prevalence = np.mean(df.values>0, axis=0)
    df = df.loc[:,cells_prevalence > min_cell_prevalence_spatial]
    
    # filter sparse cells
    genes_prevalence = np.mean(df.values>0, axis=1)
    df = df.loc[genes_prevalence > min_gene_prevalence_spatial,:]
    
    # create AnnData
    spatial_adata = ad.AnnData(X=df, dtype='float64')
    spatial_adata.obs_names = df.index.values
    spatial_adata.obs_names = spatial_adata.obs_names.astype(str)
    spatial_adata.var_names = df.columns
    del df
    
    # add spatial locations
    locations = pd.read_csv(spatial_loc,header=0,delim_whitespace=True)
    spatial_adata.obsm["spatial"] = locations.loc[genes_prevalence > min_gene_prevalence_spatial, :].values
    
    # add metadata
    if spatial_metadata is not None:
        metadata_df = pd.read_csv(spatial_metadata)
        metadata_df = metadata_df.loc[genes_prevalence > min_gene_prevalence_spatial, :]
        metadata_df.index = spatial_adata.obs_names
        spatial_adata.obs = metadata_df
    
    # remove genes with nan values
    spatial_adata = spatial_adata[:,np.isnan(spatial_adata.X).sum(axis=0)==0].copy()
    
    # make unique obs_names and var_names
    spatial_adata.obs_names_make_unique()
    spatial_adata.var_names_make_unique()
    
    return (spatial_adata)


def load_rnaseq_data (RNAseq_counts, min_cell_prevalence_RNAseq = 0.0, min_gene_prevalence_RNAseq = 0.0):
    '''
    Loads in scRNAseq data from text files.
    
    See load_paired_datasets() for details on arguments
    '''
    # read in RNAseq counts
    df = pd.read_csv(RNAseq_counts,header=0,index_col=0,sep="\t")
    
    # filter lowly expressed genes -- note that df is transposed gene x cell
    cells_prevalence = np.mean(df>0, axis=1)
    df = df.loc[cells_prevalence > min_cell_prevalence_RNAseq,:]
    del cells_prevalence
    
    # filter sparse cells
    genes_prevalence = np.mean(df>0, axis=0)
    df = df.loc[:,genes_prevalence > min_gene_prevalence_RNAseq]
    del genes_prevalence
    
    # create AnnData
    RNAseq_adata = ad.AnnData(X=df.T, dtype='float64')
    RNAseq_adata.obs_names = df.T.index.values
    RNAseq_adata.var_names = df.T.columns
    del df
    
    # remove genes with nan values
    RNAseq_adata = RNAseq_adata[:,np.isnan(RNAseq_adata.X).sum(axis=0)==0].copy()
    
    # make unique obs_names and var_names
    RNAseq_adata.obs_names_make_unique()
    RNAseq_adata.var_names_make_unique()
    
    return (RNAseq_adata)



def preprocess_data (adata, standardize=False, normalize=False):
    '''
    Preprocesses adata inplace:
        1. sc.pp.normalize_total() if normalize is True
        2. sc.pp.log1p() if normalize is True
        3. Not recommended: standardize each gene (subtract mean, divide by standard deviation)
    
    Parameters
    ----------
        standardize [Boolean] - whether to standardize genes; default is False
        normalize [Boolean] - whether to normalize data; default is False (based on finding by Li et al., 2022)
    
    Returns
    -------
        Modifies adata in-place
    
    NOTE: Under current default settings for TISSUE, this method does nothing to adata
    '''
    # normalize data
    if normalize is True:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
    
    # standardize data
    if standardize is True:
        adata.X = np.divide(adata.X - np.mean(adata.X, axis=0), np.std(adata.X, axis=0))


def build_spatial_graph (adata, method="fixed_radius", spatial="spatial", radius=None, n_neighbors=20, set_diag=True):
    '''
    Builds a spatial graph from AnnData according to specifications. Uses Squidpy implementations for building spatial graphs.
    
    Parameters
    ----------
        adata [AnnData] - spatial data, must include adata.obsm[spatial]
        method [str]:
            - "radius" (all cells within radius are neighbors)
            - "delaunay" (triangulation)
            - "delaunay_radius" (triangulation with pruning by max radius; DEFAULT)
            - "fixed" (the k-nearest cells are neighbors determined by n_neighbors)
            - "fixed_radius" (knn by n_neighbors with pruning by max radius)
        spatial [str] - column name for adata.obsm to retrieve spatial coordinates
        radius [None or float/int] - radius around cell centers for which to detect neighbor cells; defaults to Q3+1.5*IQR of delaunay (or fixed for fixed_radius) neighbor distances
        n_neighbors [None or int] - number of neighbors to get for each cell (if method is "fixed" or "fixed_radius" or "radius_fixed"); defaults to 20
        set_diag [True or False] - whether to have diagonal of 1 in adjacency (before normalization); False is identical to theory and True is more robust; defaults to True
    
    Returns
    -------
        Modifies adata in-place
    '''
    # delaunay graph
    if method == "delaunay": # triangulation only
        sq.gr.spatial_neighbors(adata, delaunay=True, coord_type="generic", set_diag=set_diag)
    
    # neighborhoods determined by fixed radius
    elif method == "radius":
        if radius is None: # compute 90th percentile of delaunay triangulation as default radius
            sq.gr.spatial_neighbors(adata, delaunay=True, coord_type="generic")
            if isinstance(adata.obsp["spatial_distances"],np.ndarray):
                dists = adata.obsp['spatial_distances'].flatten()[adata.obsp['spatial_distances'].flatten() > 0]
            else:
                dists = adata.obsp['spatial_distances'].toarray().flatten()[adata.obsp['spatial_distances'].toarray().flatten() > 0]
            radius = np.percentile(dists, 75) + 1.5*(np.percentile(dists, 75) - np.percentile(dists, 25))
        # build graph
        sq.gr.spatial_neighbors(adata, radius=radius, coord_type="generic", set_diag=set_diag)
    
    # delaunay graph with removal of outlier edges with distance > radius
    elif method == "delaunay_radius":
        # build initial graph
        sq.gr.spatial_neighbors(adata, delaunay=True, coord_type="generic", set_diag=set_diag)
        if radius is None: # compute default radius as 75th percentile + 1.5*IQR
            if isinstance(adata.obsp["spatial_distances"],np.ndarray):
                dists = adata.obsp['spatial_distances'].flatten()[adata.obsp['spatial_distances'].flatten() > 0]
            else:
                dists = adata.obsp['spatial_distances'].toarray().flatten()[adata.obsp['spatial_distances'].toarray().flatten() > 0]
            radius = np.percentile(dists, 75) + 1.5*(np.percentile(dists, 75) - np.percentile(dists, 25))
        # prune edges by radius
        adata.obsp['spatial_connectivities'][adata.obsp['spatial_distances']>radius] = 0
        adata.obsp['spatial_distances'][adata.obsp['spatial_distances']>radius] = 0
    
    # fixed neighborhood size with removal of outlier edges with distance > radius
    elif method == "fixed_radius":
        # build initial graph
        sq.gr.spatial_neighbors(adata, n_neighs=n_neighbors, coord_type="generic", set_diag=set_diag)
        if radius is None: # compute default radius as 75th percentile + 1.5*IQR
            if isinstance(adata.obsp["spatial_distances"],np.ndarray):
                dists = adata.obsp['spatial_distances'].flatten()[adata.obsp['spatial_distances'].flatten() > 0]
            else:
                dists = adata.obsp['spatial_distances'].toarray().flatten()[adata.obsp['spatial_distances'].toarray().flatten() > 0]
            radius = np.percentile(dists, 75) + 1.5*(np.percentile(dists, 75) - np.percentile(dists, 25))
        # prune edges by radius
        adata.obsp['spatial_connectivities'][adata.obsp['spatial_distances']>radius] = 0
        adata.obsp['spatial_distances'][adata.obsp['spatial_distances']>radius] = 0
            
    # fixed neighborhood size
    elif method == "fixed":
        sq.gr.spatial_neighbors(adata, n_neighs=n_neighbors, coord_type="generic", set_diag=set_diag)
            
    else:
        raise Exception ("method not recognized")
        
        
def calc_adjacency_weights (adata, method="cosine", beta=0.0, confidence=None):
    '''
    Creates a normalized adjacency matrix containing edges weights for spatial graph
        adata [AnnData] = spatial data, must include adata.obsp['spatial_connectivities'] and adata.obsp['spatial_distances']
        method [str] = "binary" (weight is binary - 1 if edge exists, 0 otherwise); "cluster" (one weight for same-cluster and different weight for diff-cluster neighbors); "cosine" (weight based on cosine similarity between neighbor gene expressions)
        beta [float] = only used when method is "cluster"; between 0 and 1; specifies the non-same-cluster edge weight relative to 1 (for same cluster edge weight)
        confidence [None or str] = if [str], then will weight edges with respective node confidences using calc_confidence_weights()
            - confidence [str] is the key for adata.obsm[confidence] for the predicted expression of confidence genes from predict_gene_expression()
    
    Adds adata.obsp["S"]:
        S [numpy matrix] = normalized weight adjacency matrix; nxn where n is number of cells in adata
    '''
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import normalize
    
    # adjacency matrix from adata
    if isinstance(adata.obsp["spatial_connectivities"],np.ndarray):
        A = adata.obsp['spatial_connectivities'].copy()
    else:
        A = adata.obsp['spatial_connectivities'].toarray().copy()
    
    # compute weights
    if method == "binary":
        pass
    
    elif method == "cluster":
        # cluster AnnData if not already clustered
        if "cluster" not in adata.obs.columns:
            sc.tl.pca(adata)
            sc.pp.neighbors(adata, n_pcs=15)
            sc.tl.leiden(adata, key_added = "cluster")
        # init same and diff masks
        cluster_ids = adata.obs['cluster'].values
        same_mask = np.zeros(A.shape)
        for i in range(A.shape[1]):
            same_mask[:,i] = [1 if cid==cluster_ids[i] else 0 for cid in cluster_ids]
        diff_mask = np.abs(same_mask-1)
        # construct cluster-based adjacency matrix
        A = A*same_mask + A*diff_mask*beta
    
    elif method == "cosine":
        # PCA reduced space
        scaler = StandardScaler()
        pca = PCA(n_components=5, svd_solver='full')
        if isinstance(adata.X,np.ndarray):
            pcs = pca.fit_transform(scaler.fit_transform(adata.X))
        else:
            pcs = pca.fit_transform(scaler.fit_transform(adata.X.toarray()))
        # cosine similarities
        cos_sim = cosine_similarity(pcs)
        # update adjacency matrix
        A = A*cos_sim
        A[A < 0] = 0
    
    else:
        raise Exception ("weighting must be 'binary', 'cluster', 'cosine'")
    
    # Compute confidence weights if specified
    if confidence is None:
        pass
    else:
        calc_confidence_weights(adata, confidence)
        A = A*adata.obs["confidence_score"].values # row-wise multiplication
        A[A < 0] = 0
    
    # normalized adjacency matrix
    S = normalize(A, norm='l1', axis=1)
    
    # update adata
    adata.obsp["S"] = S
    
    
def calc_confidence_weights (adata, confidence):
    '''
    Computes a confidence score for each cell in adata and stores it in adata.obs["confidence_score"]:
        adata - AnnData object that must have adata.uns["conf_genes_used"] (from predict_gene_expression())
        confidence - adata.obsm[confidence] contains the predicted expressions to compute scores for
    
    The score is computed as 
    '''    
    # retrieve predictions and labels of confidence genes
    confidence_predictions = adata.obsm[confidence][adata.uns["conf_genes_used"]].values
    confidence_labels = adata[:, adata.uns["conf_genes_used"]].X
    
    # compute confidence score
    confidence_score = []
    for gi in range(confidence_predictions.shape[0]):
        cos_sim = np.dot(confidence_predictions[gi,:], confidence_labels[gi,:]) / (np.linalg.norm(confidence_predictions[gi,:]) * np.linalg.norm(confidence_labels[gi,:]))
        confidence_score.append(cos_sim)
    
    adata.obs["confidence_score"] = np.nan_to_num(confidence_score, nan=0.0)


def predict_gene_expression (spatial_adata, RNAseq_adata,
                             target_genes, conf_genes=None,
                             method="spage", n_folds=None, random_seed=444, **kwargs):
    '''
    Leverages one of several methods to predict spatial gene expression from a paired spatial and scRNAseq dataset
    
    Parameters
    ----------
        spatial_adata [AnnData] = spatial data
        RNAseq_adata [AnnData] = RNAseq data, RNAseq_adata.var_names should be superset of spatial_adata.var_names
        target_genes [list of str] = genes to predict spatial expression for; must be a subset of RNAseq_adata.var_names
        conf_genes [list of str] = genes in spatial_adata.var_names to use for confidence measures; Default is to use all genes in spatial_adata.var_names
        method [str] = baseline imputation method
            "knn" (uses average of k-nearest neighbors in RNAseq data on Harmony joint space)
            "spage" (SpaGE imputation by Abdelaal et al., 2020)
            "tangram" (Tangram cell positioning by Biancalani et al., 2021)
            Others TBD
        n_folds [None or int] = number of cv folds to use for conf_genes, cannot exceed number of conf_genes, None is keeping each gene in its own fold
        random_seed [int] = used to see n_folds choice (defaults to 444)
    
    Returns
    -------
        Adds to adata the [numpy matrix]: spatial_adata.obsm["predicted_expression"], spatial_adata.obsm["combined_loo_expression"]
            - matrix of predicted gene expressions (same number of rows as spatial_adata, columns are target_genes)
    '''
    # change all genes to lower
    target_genes = [t.lower() for t in target_genes]
    spatial_adata.var_names = [v.lower() for v in spatial_adata.var_names]
    RNAseq_adata.var_names = [v.lower() for v in RNAseq_adata.var_names]
    
    # drop duplicates if any (happens in Dataset14)
    if RNAseq_adata.var_names.duplicated().sum() > 0:
        RNAseq_adata = RNAseq_adata[:,~RNAseq_adata.var_names.duplicated()].copy()
    if spatial_adata.var_names.duplicated().sum() > 0:
        spatial_adata = spatial_adata[:,~spatial_adata.var_names.duplicated()].copy()
    
    # raise warning if any target_genes in spatial data already
    if any(x in target_genes for x in spatial_adata.var_names):
        warnings.warn("Some target_genes are already measured in the spatial_adata object!")
    
    # First pass over all genes using specified method
    if method == "knn":
        predicted_expression_target = knn_impute(spatial_adata,RNAseq_adata,genes_to_predict=target_genes,**kwargs)
    elif method == "spage":
        predicted_expression_target = spage_impute(spatial_adata,RNAseq_adata,genes_to_predict=target_genes,**kwargs)
    elif method == "tangram":
        predicted_expression_target = tangram_impute(spatial_adata,RNAseq_adata,genes_to_predict=target_genes,**kwargs)
    else:
        raise Exception ("method not recognized")
        
    # Second pass over conf_genes using specified method using cross-validation
    
    if conf_genes is None:
        conf_genes = list(spatial_adata.var_names)
    conf_genes = [c.lower() for c in conf_genes]
    conf_genes_unique = [c for c in conf_genes if c not in target_genes] # removes any conf_genes also in target_genes
    if len(conf_genes_unique) < len(conf_genes):
        print("Found "+str(len(conf_genes)-len(conf_genes_unique))+" duplicate conf_gene in target_genes.")
    conf_genes_RNA = [c for c in conf_genes_unique if c in RNAseq_adata.var_names] # remove any conf genes not in RNAseq
    if len(conf_genes_RNA) < len(conf_genes_unique):
        print("Found "+str(len(conf_genes_unique)-len(conf_genes_RNA))+" conf_gene not in RNAseq_adata.")
    conf_genes = conf_genes_RNA
    
    # raise error if no conf_genes
    if len(conf_genes) == 0:
        raise Exception ("No suitable conf_genes specified!")
    
    # create folds if needed
    if n_folds is None:
        n_folds = len(conf_genes)
    elif n_folds > len(conf_genes):
        raise Warning ("n_folds in predict_gene_expression() is greater than length of conf_genes...")
        n_folds = len(conf_genes)

    np.random.seed(random_seed)
    np.random.shuffle(conf_genes)
    folds = np.array_split(conf_genes, n_folds)
    
    # run prediction on each fold
    for gi, fold in enumerate(folds):
        if method == "knn":
            loo_expression = knn_impute(spatial_adata[:,~spatial_adata.var_names.isin(fold)],RNAseq_adata,genes_to_predict=list(fold)+target_genes,**kwargs)
        elif method == "spage":
            loo_expression = spage_impute(spatial_adata[:,~spatial_adata.var_names.isin(fold)],RNAseq_adata,genes_to_predict=list(fold)+target_genes,**kwargs)
        elif method == "tangram":
            loo_expression = tangram_impute(spatial_adata[:,~spatial_adata.var_names.isin(fold)],RNAseq_adata,genes_to_predict=list(fold)+target_genes,**kwargs)
        else:
            raise Exception ("method not recognized")
    
        # Update 
        if gi == 0:
            predicted_expression_conf = loo_expression.copy()
        else:
            predicted_expression_conf['index'] = range(predicted_expression_conf.shape[0])
            loo_expression['index'] = range(loo_expression.shape[0])
            predicted_expression_conf.set_index('index')
            loo_expression.set_index('index')
            predicted_expression_conf = pd.concat((predicted_expression_conf,loo_expression)).groupby(by="index").sum().reset_index().drop(columns=['index'])
    
    # Take average of target_genes (later overwritten by "all genes"-predicted)
    predicted_expression_conf[target_genes] = predicted_expression_conf[target_genes]/(len(conf_genes))
    
    # Update spatial_adata
    predicted_expression_target.index = spatial_adata.obs_names
    predicted_expression_conf.index = spatial_adata.obs_names

    # gets predictions for target genes followed by conf genes
    predicted_expression_target[conf_genes] = predicted_expression_conf[conf_genes].copy()
    spatial_adata.obsm[method+"_predicted_expression"] = predicted_expression_target
    
    spatial_adata.uns["conf_genes_used"] = conf_genes
    spatial_adata.uns["target_genes_used"] = target_genes


def knn_impute (spatial_adata, RNAseq_adata, genes_to_predict, n_neighbors, **kwargs):
    '''
    Runs basic kNN imputation using Harmony subspace
    '''
    from scanpy.external.pp import harmony_integrate
    from scipy.spatial.distance import cdist
    
    # combine anndatas
    intersection = np.intersect1d(spatial_adata.var_names, RNAseq_adata.var_names)
    subRNA = RNAseq_adata[:, intersection]
    subspatial = spatial_adata[:, intersection]
    joint_adata = ad.AnnData(X=np.vstack((subRNA.X,subspatial.X)))
    joint_adata.obs_names = np.concatenate((subRNA.obs_names.values,subspatial.obs_names.values))
    joint_adata.var_names = subspatial.var_names.values
    joint_adata.obs["batch"] = ["rna"]*len(subRNA.obs_names.values)+["spatial"]*len(spatial_adata.obs_names.values)
    
    # run Harmony
    sc.tl.pca(joint_adata)
    harmony_integrate(joint_adata, 'batch', verbose=False)
    
    # kNN imputation
    knn_mat = cdist(joint_adata[joint_adata.obs["batch"] == "spatial"].obsm['X_pca_harmony'][:,:np.min([30,joint_adata.obsm['X_pca_harmony'].shape[1]])],
                     joint_adata[joint_adata.obs["batch"] == "rna"].obsm['X_pca_harmony'][:,:np.min([30,joint_adata.obsm['X_pca_harmony'].shape[1]])])
    k_dist_threshold = np.sort(knn_mat)[:, n_neighbors-1]
    knn_mat[knn_mat > k_dist_threshold[:,np.newaxis]] = 0 # sets all dist > thresh to 0
    knn_mat[knn_mat > 0] = 1 # 1 for connection to a nn
    row_sums = knn_mat.sum(axis=1)
    knn_mat = knn_mat / row_sums[:,np.newaxis]
    predicted_expression = knn_mat @ RNAseq_adata.X
    
    predicted_expression = pd.DataFrame(predicted_expression, columns=RNAseq_adata.var_names.values)
    predicted_expression = predicted_expression[genes_to_predict]
    
    return(predicted_expression)
    
    
def spage_impute (spatial_adata, RNAseq_adata, genes_to_predict, **kwargs):
    '''
    Runs SpaGE gene imputation
    '''
    #sys.path.append("Extenrnal/SpaGE-master/")
    from .SpaGE.main import SpaGE
    
    # transform adata in spage input data format
    if isinstance(spatial_adata.X,np.ndarray):
        spatial_data = pd.DataFrame(spatial_adata.X.T)
    else:
        spatial_data = pd.DataFrame(spatial_adata.X.T.toarray())
    spatial_data.index = spatial_adata.var_names.values
    if isinstance(RNAseq_adata.X,np.ndarray): # convert to array if needed
        RNAseq_data = pd.DataFrame(RNAseq_adata.X.T)
    else:
        RNAseq_data = pd.DataFrame(RNAseq_adata.X.T.toarray())
    RNAseq_data.index = RNAseq_adata.var_names.values
    # predict with SpaGE
    predicted_expression = SpaGE(spatial_data.T,RNAseq_data.T,genes_to_predict=genes_to_predict,**kwargs)
    
    return(predicted_expression)


def tangram_impute (spatial_adata, RNAseq_adata, genes_to_predict, **kwargs):
    '''
    Run Tangram gene imputation (positioning) using the more efficient cluster-level approach with Leiden clustering
    '''
    import torch
    from torch.nn.functional import softmax, cosine_similarity, sigmoid
    import tangram as tg
    
    # clustering and preprocessing
    RNAseq_adata_label = RNAseq_adata.copy()
    sc.pp.highly_variable_genes(RNAseq_adata_label)
    RNAseq_adata_label = RNAseq_adata[:, RNAseq_adata_label.var.highly_variable].copy()
    sc.pp.scale(RNAseq_adata_label, max_value=10)
    sc.tl.pca(RNAseq_adata_label)
    sc.pp.neighbors(RNAseq_adata_label)
    sc.tl.leiden(RNAseq_adata_label, resolution = 0.5)
    RNAseq_adata.obs['leiden'] = RNAseq_adata_label.obs.leiden
    del RNAseq_adata_label
    tg.pp_adatas(RNAseq_adata, spatial_adata) # genes=None default using all genes shared between two data
    
    # gene projection onto spatial
    ad_map = tg.map_cells_to_space(RNAseq_adata, spatial_adata, mode='clusters', cluster_label='leiden', density_prior='rna_count_based', verbose=False)
    ad_ge = tg.project_genes(ad_map, RNAseq_adata, cluster_label='leiden')
    predicted_expression = pd.DataFrame(ad_ge[:,genes_to_predict].X, index=ad_ge[:,genes_to_predict].obs_names, columns=ad_ge[:,genes_to_predict].var_names)
    
    return(predicted_expression)


def update (X, Xt, S, alpha):
    '''
    Update equation shared by reinforce() and smooth()
    '''
    Xt1 = (1-alpha)*X + alpha*(S@Xt)
    return(Xt1)
    
    
def propagate (X, S, alpha, update_method, npcs, max_iter, tol, scale=None, conf_bool=None, verbose=True):
    '''
    Iterate update() until convergence is reached. See reinforce() and smooth() for usage/argument details
        X is the numpy matrix of node values to propagate
        S is the adjacency matrix
    
    scale and conf_bool should be set to non-None values for reinforce()
    
    verbose = whether to print propagation iterations
    '''
    # independent updates
    if update_method == "joint":
        Xt = X.copy()
        Xt1 = update(X, Xt, S, alpha)

        iter_num = 1
        while (iter_num < max_iter) and np.any(np.divide(np.abs(Xt1-Xt), np.abs(Xt), out=np.full(Xt.shape,0.0), where=Xt!=0) > tol):
            Xt = Xt1
            Xt1 = update(X, Xt, S, alpha)
            iter_num += 1

    # updates on the PCA embedding space
    elif update_method == "joint_pca":
        if npcs is None:
            npcs = np.max([1,round(X.shape[1]*0.25)])
        scaler = StandardScaler()
        pca = PCA(n_components=npcs)
        pcaX = pca.fit_transform(scaler.fit_transform(X))

        pcaXt = pcaX.copy()
        pcaXt1 = update(pcaX, pcaXt, S, alpha)

        iter_num = 1
        while (iter_num < max_iter) and np.any(np.divide(np.abs(pcaXt1-pcaXt), np.abs(pcaXt), out=np.full(pcaXt.shape,0.0), where=pcaXt!=0) > tol):
            pcaXt = pcaXt1
            pcaXt1 = update(pcaX, pcaXt, S, alpha)
            iter_num += 1

        Xt1 = scaler.inverse_transform(pca.inverse_transform(pcaXt1))
    
    else:
        raise Exception ("update_method not recognized")
    
    if verbose is True:
        print("Propagation converged after "+str(iter_num)+" iterations")
    
    # autoscaling
    if scale is not None:
        Xt1 = scaling_update(X, Xt1, conf_bool, scale)
    
    return(Xt1)


def select_alpha(X, Y, S, alpha, update_method, npcs, max_iter, tol, scale=None, conf_bool=None, metric="MAE", save=False,
                 return_values=False, return_cv=False):
    '''
    Choose an optimal choice from alpha [list of float] based on minimizing the designated metric between the propagated X and Y
        X is the numpy matrix of node values to propagate
        S is the adjacency matrix
        alpha is list of alpha values to optimize over
        metric [str] is the metric to optimize (MAE, PCC, MAEImprovement, PCCImprovement)
        save [str] = False or filepath to save a CSV of results
        return_values [bool] = whether to return the full loss/values instead of the optimal alpha
        return_cv [bool] = whether to return X after propagation as the second element in a tuple
    See propagate(), reinforce(), smooth() for details on other arguments and use cases
    
    Returns:
        alpha_opt [float] = optimal alpha value selected from alpha list;
                            if return_values is True, returns list of all values across alpha instead of just alpha_opt
    '''
    # iterate over choices of alpha and compute metric
    values = []
    
    # saving additional metrics outside of main one (e.g. for "Ensemble" metric)
    additional_to_save = {}
    Xt1_list = []
    
    for a in alpha:
    
        Xt1 = propagate(X, S, a, update_method, npcs, max_iter, tol, scale, conf_bool, verbose=False)
        Xt1_list.append(Xt1)
        
        if metric == "MAE":
            values.append(np.nanmean(np.abs(Xt1-Y)))
        
        elif metric == "PCC":
            pearsons = []
            for j in range(X.shape[1]):
                if np.isnan(Y[:,j]).all(): # skip if labels are all nan
                    continue
                pearsons.append(pearsonr(Xt1[:,j],Y[:,j])[0])
            values.append(-np.nanmean(pearsons))
        
        elif metric == "MAEImprovement":
            maes = []
            maes_baseline = []
            for j in range(X.shape[1]):
                if np.isnan(Y[:,j]).all(): # skip if labels are all nan
                    continue
                maes.append(np.nanmean(np.abs(Xt1[:,j]-Y[:,j])))
                maes_baseline.append(np.nanmean(np.abs(X[:,j]-Y[:,j])))
            scores = np.array(maes) - np.array(maes_baseline)
            values.append(np.nanmean(scores))
            
        elif metric == "PCCImprovement":
            pearsons = []
            pearsons_baseline = []
            for j in range(X.shape[1]):
                if np.isnan(Y[:,j]).all(): # skip if labels are all nan
                    continue
                pearsons.append(pearsonr(Xt1[:,j],Y[:,j])[0])
                pearsons_baseline.append(pearsonr(X[:,j],Y[:,j])[0])
            scores = np.array(pearsons_baseline) - np.array(pearsons)
            values.append(np.nanmean(scores))
            
        elif metric == "Ensemble":
            maes = []
            maes_baseline = []
            pearsons = []
            pearsons_baseline = []
            for j in range(X.shape[1]):
                if np.isnan(Y[:,j]).all(): # skip if labels are all nan
                    continue
                maes.append(np.nanmean(np.abs(Xt1[:,j]-Y[:,j])))
                maes_baseline.append(np.nanmean(np.abs(X[:,j]-Y[:,j])))
                pearsons.append(pearsonr(Xt1[:,j],Y[:,j])[0])
                pearsons_baseline.append(pearsonr(X[:,j],Y[:,j])[0])
            scores_mae = np.array(maes) - np.array(maes_baseline)
            scores_pearson = np.array(pearsons_baseline) - np.array(pearsons)
            
            for key in ["MAE", "MAEImprovement", "PCC", "PCCImprovement"]:
                if key not in list(additional_to_save.keys()):
                    additional_to_save[key] = []
            
            additional_to_save["MAE"].append(np.nanmean(maes))
            additional_to_save["MAEImprovement"].append(np.nanmean(scores_mae))
            additional_to_save["PCC"].append(-np.nanmean(pearsons))
            additional_to_save["PCCImprovement"].append(np.nanmean(scores_pearson))
    
    if metric == "Ensemble":
        # get rankings and ensemble score (average rank)
        ranks_list = []
        for key in list(additional_to_save.keys()):
            ranks_list.append(np.array(additional_to_save[key]).argsort().argsort())
        scores = np.vstack(ranks_list)
        values = list(np.nanmean(scores, axis=0))
    
    # save results
    if save is not False:
        df = pd.DataFrame([])
        df['alpha'] = alpha
        df[metric] = values
        for key in list(additional_to_save.keys()):
            df[key] = additional_to_save[key]
        df.to_csv(save+"_"+metric+".csv", index=False)
    
    if return_values is False:
        # select optimal alpha
        alpha_opt = alpha[values.index(np.nanmin(values))]
        if return_cv is True:
            return (alpha_opt, Xt1_list[values.index(np.nanmin(values))])
        else:
            return (alpha_opt)
    else:
        if return_cv is True:
            return (values, Xt1_list[values.index(np.nanmin(values))])
        else:
            return(values)


def build_gene_correlation_network(G, correlation_threshold):
    '''
    Builds a GCN to use for reinforce()
        G is a gene expression matrix
        correlation_threshold is the hard Spearman correlation cutoff for an edge; "auto" to ensure connected GCN
        See reinforce() for more details on arguments
    Returns adjacency matrix S for the GCN
    '''
    from sklearn.preprocessing import normalize
    
    # compute pairwise Spearman correlations (filling in zeros for nan correlations)
    
    # mask out uniform columns with normal noise vectors
    masked = np.all(G == G[0], axis=0) # column mask for any uniform column vectors
    if np.sum(masked) > 0:
        masked_G = G.copy()
        masked_G[:, masked] = np.random.normal(size=masked_G[:, masked].shape)
        # compute spearman correlation and infill zeros
        S = spearmanr(masked_G, axis=0)[0]
        S[:, masked] = 0
        S[masked, :] = 0
    else: # no uniform columns that need to be masked out
        S = spearmanr(G, axis=0)[0]
    
    # replace nan and neg corr with zero
    S[np.isnan(S)] = 0
    S[S<0] = 0
        
    # set automatic threshold if needed
    if correlation_threshold == "auto":
        S_no_diag = S[~np.eye(S.shape[0],dtype=bool)].reshape(S.shape[0],-1)
        max_corr_per_node = np.max(S_no_diag, axis=0)
        correlation_threshold = np.min(max_corr_per_node)
        print("Using automatic Spearman gene correlation threshold: "+str(round(correlation_threshold, 3)))
    elif isinstance(correlation_threshold, float):
        if (correlation_threshold <= 1.0) and (correlation_threshold >= 0.0):
            pass
        else:
            raise Exception("correlation_threshold must be between 0 and 1")
    else:
        raise Exception("correlation_threshold is not recognized; please use 'auto' or a float between 0 and 1")
    
    # filter by correlation threshold
    S[S<correlation_threshold] = 0
    S = normalize(S, norm='l1', axis=1)
    
    return (S)
    
    
def reinforce_gene (adata, predicted="predicted_expression", update_method="joint",
                    npcs=None, alpha=0.1, max_iter=100, tol=1e-3, scale="auto_gene", correlation_threshold="auto",
                    optimization_metric="MAE", savedir=False, cv=5):
    '''
    Reinforces predicted gene expressions using the spatial graph and precomputed reinforcment expressions from neighbors in RNAseq data
        predicted [str] = key in adata.obsm corresponding to predicted gene expression matrix; added by predict_gene_expression()
        confidence [str] = key in adata.obsm corresponding to confidence gene expression matrix; added by predict_gene_expression()
        update_method = 'joint' or 'joint_pca' (default is 'joint')
        npcs [int] = number of PCs to make updates with if method=="joint_pca"; defaults to 0.25*n_genes
        alpha [float] = learning rate for updates; between 0 and 1; if list it will cv-fold optimization to select optimal value from alpha
        max_iter [int] = maximum number of updates
        tol [float] = relative tolerance/threshold for stopping updates; at iteration t, it is compared to  |(value at t)-(value at t-1)|/|value at t-1| [does not converge if the value at t-1 is 0]
        scale [str or None] = scaling for prediction residuals: "auto" - L1 norm autoscaled to confidence residuals
        correlation_threshold [float or "auto"] = threshold for determing gene network connectivity (between 0 and 1) on Spearman correlation scale
            if "auto": the correlation threshold will be determined to ensure that all nodes/genes have at least one non-self edge
        optimization_metric [str] = for optimizing choice of alpha (only compatible with "MAE" currently)
        savedir [str] = where to save the alpha optimization results; doesn't save if False
        cv [int] = how many CV folds to use to find optimal alpha values; only used if multiple values are provided for alpha
        
    Adds adata.obsm["reinforced_expression"]:
        - matrix of same shape as adata.obsm[predicted] containing reinforced expression values
    '''
    from sklearn.preprocessing import normalize
    
    # get G and Y matrices
    G, Y, columns, conf_bool = get_prediction_and_conf_labels(adata, predicted)
    if np.any(conf_bool) is False:
        raise Exception ("No confidence genes detected, need confidence genes to perform reinforcement")
    
    # get gene correlation graph
    if "S" in adata.varp.keys():
        S = adata[:, columns].varp["S"]
    else: # build gene correlation network
        S = build_gene_correlation_network(G, correlation_threshold)
        
    # create residual matrix, leaving zeros for target genes
    E = np.zeros(G.shape)
    E[:,conf_bool] = Y[:,conf_bool]-G[:,conf_bool]
    E = E.T
        
    # optimize alpha value
    if isinstance(alpha,(list,pd.core.series.Series,np.ndarray)):
        #YE = np.zeros(G.shape).T # label for residuals will be 0-matrix
        # make CV folds
        conf_idxs = np.where(conf_bool)[0]
        np.random.seed(444)
        np.random.shuffle(conf_idxs)
        cv_folds = np.array_split(conf_idxs, cv)
        # get alpha losses for CV folds
        losses = []
        G_cv_df = pd.DataFrame([])
        for i in range(cv):
            # create cv E and YE
            E_masked = E.copy()
            YE_masked = E.copy()
            E_masked[cv_folds[i],:] = 0 # mask out the fold in E
            YE_masked[~np.isin(np.arange(YE_masked.shape[0]), cv_folds[i]),:] = np.nan # mask out non-fold with NA in YE
            # use optimization_metric = "MAE" since looking at errors and PCC not well-defined
            if savedir is False:
                loss, E_cv = select_alpha(E_masked, YE_masked, S, alpha, update_method, npcs, max_iter, tol, scale, conf_bool, metric=optimization_metric, save=False, return_values=True, return_cv=True)
            else:
                if not os.path.exists(savedir):
                    os.makedirs(savedir)
                loss, E_cv = select_alpha(E_masked, YE_masked, S, alpha, update_method, npcs, max_iter, tol, scale, conf_bool, metric=optimization_metric, save=os.path.join(savedir,predicted+"_"+str(i)), return_values=True, return_cv=True)
            losses.append(loss)
            G_cv_df[columns[cv_folds[i]]] = (G+E_cv.T)[:,cv_folds[i]]
        # average losses across folds and find optimal alpha
        avg_loss = np.nanmean(np.vstack(losses), axis=0)
        alpha = alpha[list(avg_loss).index(np.nanmin(avg_loss))]
        print("Selected alpha = "+str(alpha)+" for reinforce")
        # add G_cv_df to adata.obsm
        G_cv_df.index = adata.obsm[predicted].index
        adata.obsm["reinforced_gene_"+update_method+"_"+predicted+"_cv_conf"] = G_cv_df
    
    # propagate residuals
    Et1 = propagate(E, S, alpha, update_method, npcs, max_iter, tol, scale, conf_bool)
    
    # reinforce G
    G_reinforced = G + Et1.T
        
    # update adata
    reinforced_expression = pd.DataFrame(G_reinforced, columns=columns)
    reinforced_expression = reinforced_expression[adata.obsm[predicted].columns]
    reinforced_expression.index = adata.obsm[predicted].index
    adata.obsm["reinforced_gene_"+update_method+"_"+predicted] = reinforced_expression


def scaling_update (E, Et1, conf_bool, scale):
    '''
    Scales the updated residuals using the original residuals
    
    Inputs:
        E (rows are cells, columns are genes) is array of initial residuals
        Et1 is same size array as E and corresponds to updated residuals
        conf_bool is a boolean array indicating whether a gene in E is a confidence gene
        scale is a string determining the type of scaling (currently only 'auto' is supported)
        
    Returns: scaled Et1
    '''
    if scale == "auto":
        E_conf = E[:,conf_bool].copy()
        l1_conf = np.mean(np.abs(E_conf), axis=1)
        l1_pred = np.mean(np.abs(Et1), axis=1)
        Et1 = np.divide(Et1,(l1_pred/l1_conf)[:,None])
    elif scale == "auto_gene":
        E_conf = E[conf_bool,:].copy()
        l1_conf = np.mean(np.abs(E_conf), axis=0)
        l1_pred = np.mean(np.abs(Et1), axis=0)
        Et1 = np.divide(Et1,(l1_pred/l1_conf)[None,:])
    else:
        raise Exception ("scale not recognized")
            
    return (Et1)
    
    
def get_prediction_and_conf_labels (adata, predicted):
    '''
    adata is the AnnData object
    predicted [str] is the key in adata.obsm for which G corresponds to
    
    Takes in AnnData adata with predicted [str] and returns the arrays:
        G = predicted expression of all confidence and target genes
        Y = reinforcement expression of all confidence genes (actual values in adata.X) and target (nan)
        columns = list of [str] corresponding to the order of genes in the columns of G and Y
    '''
    # get conf gene info
    conf_genes = adata.uns['conf_genes_used'].copy()
    columns = adata.obsm[predicted].columns.copy()
    conf_bool = np.isin(columns, conf_genes)
    
    # make G
    G = adata.obsm[predicted].values.copy() # gene expression array
    
    # make Y
    Y = np.ones(G.shape)*np.nan # init array of zeros
    Y[:, conf_bool] = adata[:, columns[conf_bool]].X.copy()
    
    return(G, Y, columns, conf_bool)
    
    
def smooth (adata, predicted=None, update_method="joint",
            npcs=None, alpha=0.1, max_iter=100, tol=1e-3,
            optimization_metric="Ensemble", savedir=False):
    '''
    Smooths predicted gene expressions using the spatial graph
        predicted [str or None] = key in adata.obsm corresponding to predicted gene expression matrix; added by predict_gene_expression()
                          if predicted is None, then smoothing will be done on adata.X and saved as adata.obsm["presmoothed_X"]
        update_method [str] = method for reinforcement: "joint" (make joint updates on gene space); "joint_pca" (make joint updates on pca space)
        npcs [int] = number of PCs to make updates with if method=="joint_pca"
        alpha [float] = learning rate for updates; between 0 and 1; if list it will cv-fold optimization to select optimal value from alpha
        max_iter [int] = maximum number of updates
        tol [float] = relative tolerance/threshold for stopping updates; at iteration t, it is compared to |(value at t)-(value at t-1)|/(value at t-1) [does not converge if the value at t-1 is 0]
        optimization_metric [str] = for optimizing choice of alpha (Ensemble, MAE, PCC, MAEImprovement, PCCImprovement)
        savedir [str] = where to save the alpha optimization results; doesn't save if False
        
    Adds adata.obsm["smoothed_expression"]:
        - matrix of same shape as adata.obsm[predicted] containing smoothed expression values
    '''
    # get predicted gene expressions
    if predicted is None:
        G = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
        columns = G.columns
        indices = G.index
        G = G.values
    else:
        G, Y, columns, conf_bool = get_prediction_and_conf_labels(adata, predicted)
        indices = adata.obsm[predicted].index
    
    # get adjacency matrix
    S = adata.obsp["S"]
    
    # optimize alpha value
    if isinstance(alpha,(list,pd.core.series.Series,np.ndarray)):
        if "reinforce" in predicted:
            if predicted+"_cv_conf" in adata.obsm.keys():
                print("Selecting smooth alpha using CV prediction from '"+predicted+"_cv_conf"+"'")
                G[:, conf_bool] = adata.obsm[predicted+"_cv_conf"][columns[conf_bool]].values # replace the reinforced confidence genes with values from CV
        if savedir is False:
            alpha = select_alpha(G, Y, S, alpha, update_method, npcs, max_iter, tol, metric=optimization_metric, save=False)
        else:
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            alpha = select_alpha(G, Y, S, alpha, update_method, npcs, max_iter, tol, metric=optimization_metric, save=os.path.join(savedir,predicted))
        print("Selected alpha = "+str(alpha)+" for smooth")
    
    # smooth predictions
    Gt1 = propagate(G, S, alpha, update_method, npcs, max_iter, tol)
    
    # update adata
    if predicted is None:
        adata.obsm["presmoothed_X"] = pd.DataFrame(Gt1, columns=columns, index=indices)
    else:
        adata.obsm["smoothed_"+predicted] = pd.DataFrame(Gt1, columns=columns, index=indices)
    
    
def large_save(adata, dirpath):
    '''
    Saves anndata objects by saving each obsm value with its {key}.csv as pandas dataframe
    Saves each uns value that is a dataframe with uns/{key}.csv as pandas dataframe
    Then saves the anndata object with obsm removed.
    
    Parameters
    ----------
        adata [AnnData] - AnnData object to save
        
        dirpath [str] - path to directory for where to save the h5ad and csv files; will create if not existing
            adata will be saved as {dirpath}/adata.h5ad
            obsm will be saved as {dirpath}/{key}.csv
        
    Returns
    -------
        Saves anndata object in "large" folder format
    '''
    # check if dirpath exists; else create it
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    
    # extract the obsm metadata and save it as separate csv files
    for key, value in adata.obsm.items():
        df = pd.DataFrame(value)
        df.to_csv(os.path.join(dirpath, f"{key}.csv"), index=False)

    # remove the obsm metadata from the anndata object
    adatac = adata.copy()
    adatac.obsm = {}
    
    # extract the uns metadata and save it as separate csv files
    del_keys = []
    for key, value in adatac.uns.items():
        if isinstance(value, pd.DataFrame):
            if not os.path.exists(os.path.join(dirpath,"uns")):
                os.makedirs(os.path.join(dirpath,"uns"))
            df = pd.DataFrame(value)
            df.to_csv(os.path.join(dirpath,"uns",f"{key}.csv"), index=False)
            del_keys.append(key)
    
    # remove uns metadata from the anndata object
    for key in del_keys:
        del adatac.uns[key]

    # save the new anndata object
    adatac.write(os.path.join(dirpath, "adata.h5ad"))



def large_load(dirpath, skipfiles=[]):
    '''
    Loads in anndata and associated pandas dataframe csv files to be added to obsm metadata and uns metadata.
    Input is the directory path to the output directory of large_save()
    
    Parameters
    ----------
        dirpath [str] - path to directory for where outputs of large_save() are located
        skipfiles [list] - list of filenames to exclude from anndata object
    
    Returns
    -------
        adata - AnnData object loaded from dirpath along with all obsm and uns key values added to metadata
    '''
    # read h5ad anndata object
    adata = ad.read_h5ad(os.path.join(dirpath, "adata.h5ad"))
    
    # read and load in obsm from CSV files
    for fn in os.listdir(dirpath):
        if (".csv" in fn) and (fn not in skipfiles):
            df = pd.read_csv(os.path.join(dirpath, fn))
            df.index = adata.obs_names
            key = fn.split(".")[0]
            adata.obsm[key] = df
            
    # read and load any usn metadata from CSV files
    if os.path.isdir(os.path.join(dirpath,"uns")):
        for fn in os.listdir(os.path.join(dirpath,"uns")):
            if (".csv" in fn) and (fn not in skipfiles):
                df = pd.read_csv(os.path.join(dirpath,"uns",fn))
                key = fn.split(".")[0]
                adata.uns[key] = df
            
    return(adata)