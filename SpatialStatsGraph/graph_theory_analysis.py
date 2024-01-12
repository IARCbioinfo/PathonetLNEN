import json
import pandas as pd
import os
import networkx as nx
import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from statistics import mean
from numpy import linalg
import pandas as pd
import argparse
import math
global folder_name
global root
import time
import datetime
import pathlib



def create_graph(df, max_nn_dist=102.4):
    """ Create a graph: This function uses the coordinates of the positive cells detected by pathonet to create a graph by calculating a distance matrix. 
    Two cells are linked (i.e. an edge is created) if the distance between them in terms of number of pixels is less than max_nn_dist.
    Args :
    - df (pandas data frame) : Table of cell coordinates see > table_of_cells_after_segmentation.py
    - max_nn_dist: Maximum distance in pixels to link cells.
    Return :
    - G (Graph NetworkX object): Graph of positive cells detected by Pathonet for the current patient_id/WSI.
    Note:
    - This function has a high memory cost due to the calculation of the distance matrix; create_graph_local_search should be applied if there are many positive cells.
    However, create_graph_local_search is slower than create_graph.
    """
    # Get the positive cells to the marker
    df_pos = df[df['label'] == 1]
    # Compute the distance matrix
    df_coord = pd.DataFrame(df_pos, columns=['x', 'y'])
    dist_matrix = distance_matrix(df_coord.values, df_coord.values)
    # Created a weighted adjacency matrix
    weighted_adj_matrix = np.zeros((dist_matrix.shape[0], dist_matrix.shape[1]))
    for i_cell in range(dist_matrix.shape[0]):
        # Get the distance vector for the current cell
        c_cell = dist_matrix[i_cell,:]
        # Get the index of all the neighboring cells of c_cell ie. if their distance is lower than max_nn_dist
        index_sort = np.where(c_cell<max_nn_dist)[0].tolist() 
        # Remove the current cell of the distance matrix
        index_sort.remove(i_cell)
        # Complete the weighted adjacency matrix in function of the neighbors found
        if len(index_sort)> 0:
            for ind in index_sort:
                weighted_adj_matrix[i_cell,ind] = dist_matrix[i_cell,ind]
    G = nx.from_numpy_array(weighted_adj_matrix)
    return G

def create_graph_local_search(max_dist, args):
    """ create_graph_local_search: This function uses the coordinates of the positive cells detected by pathonet
    to create a graph by searching locally for the potential neighbours of each cell. 
    The tiles are considered one by one, and for each one we define the neighbouring tiles as the eight surrounding tiles.
    For each neighbouring tile and for the current tile itself, the positive cells are listed. 
    The Euclidean distances between each cell of the current tile and those of the cells included in the neighbourhood are calculated.
    Edges are added to the graph between two cells less than max_dist pixels apart, otherwise the cell is added to the graph as an isolated cell. 
    Args :
        - max_nn_dist: Maximum distance in pixels to link cells.
    Return :
    - G (Graph NetworkX object): Graph of positive cells detected by Pathonet for the current patient_id/WSI.
    """
    # Create an empty graph
    G = nx.Graph()
    count_pos_cells = 0
    c_json_treated  = 0
    # Path to inference json file directory
    tot_to_treat = len(os.listdir(os.path.join(args['rootdir'], args['patient_id'], 'accept')))
    for c_json in os.listdir(os.path.join(args['rootdir'], args['patient_id'], 'accept')):
        f_c_json = open(os.path.join(args['rootdir'], args['patient_id'],'accept'  ,c_json ), "r")
        data_c_json = json.load(f_c_json)
        # Get tile coordinates of the current tiles 
        xul = int(c_json.split('_')[1])
        yul = int(c_json.split('_')[-1].split('.')[0])
        # Obtain a list of cells in the vicinity (9 tiles surrounding the current tile and itself).
        cells_in_neighborhood = get_neighborhood_pos_cell_table(c_json, xul, yul)

        # For every cell in the current tile
        for cell in data_c_json:
            # Define attibutes of the current cell
            c_x = cell['x'] + xul
            c_y = cell['y'] + yul
            c_cell_name = str(c_x)+ '_' + str(c_y)
            nb_neighbors_for_current_cell = 0
            # Get its coordinates
            if cell['label_id'] == 1:
                c_x = cell['x'] + xul
                c_y = cell['y'] + yul
                # Check whether the cells listed in the neighbourhood tiles and the tile itself
                # are less than max_dist from the current cell.
                for i in range(cells_in_neighborhood.shape[0]):
                    c_nx = cells_in_neighborhood.iloc[i,0]
                    c_ny = cells_in_neighborhood.iloc[i,1]
                    # Compute euclidian distance
                    d = distance_euclidienne(c_x, c_nx, c_y, c_ny)
                    # If two cells are less than max_dist apart, an edge is created between them.
                    if d < max_dist:
                        c_cell_name = str(c_x)+ '_' + str(c_y)
                        c_n_cell_name = str(c_nx)+ '_' + str(c_ny)
                        G.add_edge(c_cell_name, c_n_cell_name, weight=d)
                        nb_neighbors_for_current_cell += 1
                # If the cell has no neighbors add it to the graph as an isolated node
                if nb_neighbors_for_current_cell == 0 : # Unconnected node
                    G.add_node(c_cell_name)
        # Progression
        c_json_treated +=1 
        print((c_json_treated / tot_to_treat)*100)
    
    zero_edges = list(filter(lambda e: e[2] == 0, (e for e in G.edges.data('weight'))))
    le_ids = list(e[:2] for e in zero_edges)
    G.remove_edges_from(le_ids)
    return G

def get_neighborhood_pos_cell_table(c_json, xul, yul):
    """
    get_neighborhood_pos_cell_table: Allows to get a list of cells in the neighbouring tiles for this the 8 tiles surrounding the current tiles (c_json) 
    are considered. This hypothesis is corrected considering an circle of 2000 micron^2.
    Args:
    - c_json : Current json files containing the list of detected cells
    - xul and yul (int): - coordinated of the lower left corner of the current tile 
    Return:
    - cells_in_neighborhood (pandas data frame): Table of cells in the neighborhood
    """
    # Coordinated neibouring tiles
    x_coords = [xul-512, xul, xul+512]
    y_coords = [yul-512, yul, yul+512]
    cells_in_neighborhood = pd.DataFrame(columns=['x', 'y'])
    for x in x_coords:
        for y in y_coords:
            c_cells_in_neighborhood  = pd.DataFrame()
            # Read json
            json_name =c_json.split('_')[0] + '_' + str(x)  + '_' + str(y) + '.json'
            if os.path.exists(os.path.join(root, folder_name, 'accept', json_name)):
                f_c_json = open(os.path.join(root, folder_name,'accept'  ,json_name ), "r")
                data_c_json = json.load(f_c_json)
                # Get list of positive cells to the market
                l_x_neighbors, l_y_neighbors = get_neighbors_coords_list(data_c_json, x, y)
                c_cells_in_neighborhood['x'] = l_x_neighbors
                c_cells_in_neighborhood['y'] = l_y_neighbors
                cells_in_neighborhood = cells_in_neighborhood.append(c_cells_in_neighborhood)
    return cells_in_neighborhood

def get_neighbors_coords_list(data_c_json, x, y):
    """
    Obtain the list of cell coordinates in neighbouring tiles
    Call by: get_neighborhood_pos_cell_table
    """
    l_x_neighbors  = []
    l_y_neighbors = []
    for cell in data_c_json:
        if cell['label_id'] == 1:
            l_x_neighbors.append(x+cell['x'])
            l_y_neighbors.append(y+cell['y'])
    return l_x_neighbors, l_y_neighbors



def distance_euclidienne(c_x, c_nx, c_y, c_ny):
    return math.sqrt((c_x - c_nx) ** 2 +(c_y - c_ny)**2)

if __name__ == "__main__":
    
    ## Argumennt
    parser = argparse.ArgumentParser(description='Graph theory - compute global and local features on KI67 detected cells')
    parser.add_argument('--rootdir', type=str,  default=  '/home/mathiane/LNENWork/PathonetCombinedDataSet2/PredBreastLNENDataset2Epoch50', help="rootdir where are the TNEXXXX_cells.csv")
    parser.add_argument('--patient_id', type=str,    help='patient_id currently under analysis')
    args = vars(parser.parse_args())
    args = vars(parser.parse_args())
    root =  args['rootdir']
    folder_name = args['patient_id']
    root = args['rootdir']

    # Check if table of cells detected in the tumor area has been created
    segmentation = False
    if os.path.exists(f'{root}/{folder_name}/{patient_id_name_tneid}_cells_detected_segmented.csv'):
        df = pd.read_csv(f'{root}/{folder_name}/{patient_id_name_tneid}_cells_detected_segmented.csv')
        print(df)
        segmentation = True
    if segmentation:
        # For an area of 2000 micron^2
        # if 256 pixel ~= 12 micron
        # Since r = sqrt(A/pi)
        # r = 25.23 micron 
        # 538.24 pixel = 25.23 micron 
        
        for max_nn_dist in [538.24]: # You can add several radius in pixels to the list
            # A radius of 538.25 px correspond to a circle of area of 2000 micron^2 ~= hot spot area measured by pathologists
            max_nn_micron = '2000_micron'
            # Test if the graph has already beeen computed
            if  os.path.exists(f'{root}/{folder_name}/{patient_id_name}_{max_nn_micron}_segmentation.gpickle'):
                Graph_name =  f'{root}/{folder_name}/{patient_id_name}_{max_nn_micron}_segmentation.gpickle'
                graphs_created = True
            else:
                Graph_name =  f'{root}/{folder_name}/{patient_id_name}_{max_nn_micron}.gpickle'
                graphs_created = False
                
            # If the graph alredy exist we computed only the spatial statistics
            if graphs_created :
                print(f'{Graph_name}  alreafdy exist')
                # Read the graph
                G = nx.read_gpickle(Graph_name)
                
                if G.number_of_nodes() == 0: # Empty graph
                    print("The graph loaded is empty!")
                    try:
                        G = create_graph(df, max_nn_dist=max_nn_dist)
                        print("Write graph")
                        nx.write_gpickle(G, Graph_name)
                        graphs_created = True
                    except:
                        print('Creation of a graph by local search ')
                        G = create_graph_local_search( max_nn_dist, args)
                        print("Write graph")
                        nx.write_gpickle(G, Graph_name)
                        graphs_created = True
                        
            # The graph have never been created
            else: 
                try:
                    G = create_graph(df, max_nn_dist=max_nn_dist)
                    print("Write graph")
                    nx.write_gpickle(G, Graph_name)
                    graphs_created = True
                except:
                    print('Creation of a graph by local search ')
                    G = create_graph_local_search( max_nn_dist,  args)
                    print("Write graph")
                    nx.write_gpickle(G, Graph_name)
                    graphs_created = True
                    
            # Compute spatial metric
            if graphs_created:
                ## Get global features
                global_features = {}
                if G.number_of_nodes() >0 :
                    global_features['nb_nodes'] = G.number_of_nodes() # Number of nodes
                    global_features['nb_edges'] = G.number_of_edges() # Number of edges
                    global_features['poucent_unconnected_nodes'] =  (len(list(nx.isolates(G))) / 
                                                                          G.number_of_nodes()) *100 # Pourcentage of unconnected nodes
                    degrees = dict(nx.degree(G))
                    global_features['poucent_end_nodes'] = (len([n for n in degrees if degrees[n]  ==  1]) /
                                                                     G.number_of_nodes()) *100 # Pourcentage of end nodes
                    global_features['size_largest_cc'] = len(max(nx.connected_components(G), 
                                                                    key=len)) # Size of the largest connected component
                    CG = nx.connected_components(G)
                    global_features['avg_size_cc_norm_nb_nodes'] =  mean([len(g) for g in CG])/ G.number_of_nodes() # Connected components average size normalized by the number of nodes
                    global_features['global_efficiency'] =  nx.global_efficiency(G)
                    
                    ## Save global features statistics 
                    if segmentation:
                        json_global_feature_fname = f'{root}/{folder_name}/{patient_id_name}_graph_{max_nn_micron}_global_features_segmented.json'
                    else:
                        json_global_feature_fname = f'{root}/{folder_name}/{patient_id_name}_graph_{max_nn_micron}_global_features.json'
                    with open(json_global_feature_fname, 'w+') as f:
                        json.dump(global_features, f)
                    print('Global spatial metric written')
                    
                    # Compute local spatial statistics
                    ## Add cell coordinates as attributes to graph nodes
                    df_pos = df[df['label'] == 1]
                    print("df_pos ", df_pos.shape)
                    df_coord = pd.DataFrame(df_pos, columns=['x', 'y'])
                    xdict = {}
                    ydict = {}
                    for i in range(df_coord.shape[0]):
                        xdict[i] = df_coord.iloc[i,0]
                        ydict[i] = df_coord.iloc[i,1]
                    nx.set_node_attributes(G, xdict, "x_coord")
                    nx.set_node_attributes(G, ydict, "y_coord")

                    degrees = dict(nx.degree(G)) # Node degree
                    print('Degrees Calculated')
                    closeness_centrality =  dict(nx.closeness_centrality(G))
                    print('closeness_centrality done')
                    weighted_closeness_centrality = dict(nx.closeness_centrality(G, distance='weight'))
                    print('weighted_closeness_centrality done') 
                    pagerank_centrality = dict(nx.pagerank(G, weight= 'weight'))
                    print('pagerank_centrality  done')
                    # eigenvector_centrality_computed = False
                    # try:
                    #     eigenvector_centrality = dict(nx.eigenvector_centrality(G))
                    #     print('eigenvector_centrality done')
                    #     eigenvector_centrality_computed = True
                    # except:
                    #     print('Error with eigenvector_centrality')
                    clustering_coeff = dict(nx.clustering(G))
                    print('clustering_coeff done')
                    weighted_clustering_coeff = dict(nx.clustering(G, weight='weight'))
                    print('weighted_clustering_coeff done')

                    # Save local spatial metric in a data frame 
                    local_feature = pd.DataFrame()
                    local_feature['x_coord'] = list(xdict.values())
                    local_feature['y_coord'] = list(ydict.values())
                    local_feature['degrees'] = list(degrees.values())
                    local_feature['closeness_centrality'] = list(closeness_centrality.values())
                    local_feature['weighted_closeness_centrality'] = list(weighted_closeness_centrality.values())
                    local_feature['pagerank_centrality']  = list(pagerank_centrality.values())
                    local_feature['clustering_coeff'] = list(clustering_coeff.values())
                    local_feature['weighted_clustering_coeff'] = list(weighted_clustering_coeff.values())
                    
                    # Write local spatial metrics
                    if segmentation:
                        csv_local_feature_fname = f'{root}/{folder_name}/{patient_id_name}_graph_{max_nn_micron}_local_features_segmented.csv'
                    else:
                        csv_local_feature_fname = f'{root}/{folder_name}/{patient_id_name}_graph_{max_nn_micron}_local_features.csv'
                    local_feature.to_csv(csv_local_feature_fname,index=False)
                    print('Local spatial metrics  written')
                else:
                    print('ERROR any positive cells > Spatial statistic cannot be computed!')
        else:
            print('ERROR patient_id_name  ', patient_id_name)
    else:
        print("Table of cell detected in the tumor area not found! Run > table_of_cells_after_segmentation.py for this patient id")