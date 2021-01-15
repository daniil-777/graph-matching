import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch


def visualize_graph(image,keypoints,adj,name):
     # keypoints here is a dictionary of keypoints,i.e, {0:(x0,y0),1:(x1,y1),2:(x2,y2)}
     # adj - adjacency matrix (0/1 values) and not a probability matrix

     # if var keypoints is a list, i.e, [(x0,y0),(x1,y1),(x2,y2)], convert it to a dictionary
     # keypoints_dict = {i: keypoints[i] for i in range(len(keypoints))} 

     # if image is a not tensor but an image file, use :
     # plt.imshow(plt.imread(image_filename))

    g = nx.Graph()
    g.add_nodes_from(torch.arange(len(keypoints), dtype=torch.int32))

    for n, p in keypoints.items():
        g.nodes[n]['pos'] = p

    edges = torch.nonzero(adj)
    g.add_edges_from(edges)

    pos= nx.get_node_attributes(g,'pos')

    plt.figure(figsize=(10, 5))
    implot = implot = plt.imshow(image.permute(1, 2, 0)) # assuming image is a tensor with 3 channels (RGB)
    ax = plt.subplot(121)
    nx.draw_networkx(g, pos=pos, ax=ax, edge_width=2.0, edge_color='r', node_color='g')
    ax.set_title('Learned Graph of Image Keypoints')
    ax.axis('off')
    plt.savefig("Learned_Graph_10_"+name)
    plt.close()
