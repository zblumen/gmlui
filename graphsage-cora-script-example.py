###########################################################################
# Copyright 2018 Data61, CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###########################################################################
# Modifications Copyright 2019 Zach Blumenfeld
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Graph node classification using GraphSAGE.
This currently is only tested on the CORA dataset, which can be downloaded from https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz

The following is the description of the dataset:
> The Cora dataset consists of 2708 scientific publications classified into one of seven classes.
> The citation network consists of 5429 links. Each publication in the dataset is described by a
> 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary.
> The dictionary consists of 1433 unique words. The README file in the dataset provides more details.

Download and unzip the cora.tgz file to a location on your computer and assign to args.location
"""
import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import keras
from keras import optimizers, losses, layers, metrics
from sklearn import preprocessing, feature_extraction, model_selection
import stellargraph as sg
from stellargraph.layer import (
    GraphSAGE,
    MeanAggregator,
    MaxPoolingAggregator,
    MeanPoolingAggregator,
)
from stellargraph.mapper import GraphSAGENodeGenerator
from types import SimpleNamespace
import hashlib
import json 
import random

args = SimpleNamespace(
        checkpoint = None, #Load a saved checkpoint .h5 file
        batch_size = 20, #Batch size for training
        epochs = 10, #The number of epochs to train the model
        dropout = 0.3, #Dropout rate for the GraphSAGE model, between 0.0 and 1.0
        learningrate = 0.005, #Initial learning rate for model training
        neighbour_samples = [20, 10], #The number of neighbour nodes sampled at each GraphSAGE layer
        layer_size = [20,20],#The number of hidden features at each GraphSAGE layer
        location = "../data/cora", #Location of the CORA dataset (directory)"
        target = "subject", #The target node attribute (categorical)
        cy_json_outfile = '../data/pred_graph' #begining of cytoscape graph output, 
        )

def train(
    edgelist,
    node_data,
    layer_size,
    num_samples,
    batch_size=100,
    num_epochs=10,
    learning_rate=0.005,
    dropout=0.0,
    target_name="subject",
):
    """
    Train a GraphSAGE model on the specified graph G with given parameters, evaluate it, and save the model.

    Args:
        edgelist: Graph edgelist
        node_data: Feature and target data for nodes
        layer_size: A list of number of hidden nodes in each layer
        num_samples: Number of neighbours to sample at each layer
        batch_size: Size of batch for inference
        num_epochs: Number of epochs to train the model
        learning_rate: Initial Learning rate
        dropout: The dropout (0->1)
    """
    # Extract target and encode as a one-hot vector
    target_encoding = feature_extraction.DictVectorizer(sparse=False)
    node_targets = target_encoding.fit_transform(
        node_data[[target_name]].to_dict("records")
    )
    node_ids = node_data.index

    # Extract the feature data. These are the feature vectors that the Keras model will use as input.
    # The CORA dataset contains attributes 'w_x' that correspond to words found in that publication.
    node_features = node_data[feature_names]

    # Create graph from edgelist and set node features and node type
    Gnx = nx.from_pandas_edgelist(edgelist)

    # Convert to StellarGraph and prepare for ML
    G = sg.StellarGraph(Gnx, node_type_name="label", node_features=node_features)

    # Split nodes into train/test using stratification.
    train_nodes, test_nodes, train_targets, test_targets = model_selection.train_test_split(
        node_ids,
        node_targets,
        train_size=140,
        test_size=None,
        stratify=node_targets,
        random_state=5232,
    )

    # Split test set into test and validation
    val_nodes, test_nodes, val_targets, test_targets = model_selection.train_test_split(
        test_nodes, test_targets, train_size=500, test_size=None, random_state=5214
    )

    # Create mappers for GraphSAGE that input data from the graph to the model
    generator = GraphSAGENodeGenerator(G, batch_size, num_samples, seed=5312)
    train_gen = generator.flow(train_nodes, train_targets, shuffle=True)
    val_gen = generator.flow(val_nodes, val_targets)

    # GraphSAGE model
    model = GraphSAGE(
        layer_sizes=layer_size,
        generator=train_gen,
        bias=True,
        dropout=dropout,
        aggregator=MeanAggregator,
    )
    # Expose the input and output sockets of the model:
    x_inp, x_out = model.default_model(flatten_output=True)

    # Snap the final estimator layer to x_out
    prediction = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)

    # Create Keras model for training
    model = keras.Model(inputs=x_inp, outputs=prediction)
    model.compile(
        optimizer=optimizers.Adam(lr=learning_rate, decay=0.001),
        loss=losses.categorical_crossentropy,
        metrics=[metrics.categorical_accuracy],
    )
    print(model.summary())

    # Train model
    history = model.fit_generator(
        train_gen, epochs=num_epochs, validation_data=val_gen, verbose=2, shuffle=False
    )

    # Evaluate on test set and print metrics
    test_metrics = model.evaluate_generator(generator.flow(test_nodes, test_targets))
    print("\nTest Set Metrics:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    # Get predictions for all nodes
    all_predictions = model.predict_generator(generator.flow(node_ids))

    # Turn predictions back into the original categories
    node_predictions = pd.DataFrame(
        target_encoding.inverse_transform(all_predictions), index=node_ids
    )
    accuracy = np.mean(
        [
            "subject=" + gt_subject == p
            for gt_subject, p in zip(
                node_data["subject"], node_predictions.idxmax(axis=1)
            )
        ]
    )
    print("All-node accuracy: {:3f}".format(accuracy))

    # TODO: extract the GraphSAGE embeddings from x_out, and save/plot them

    # Save the trained model
    save_str = "_n{}_l{}_d{}_r{}".format(
        "_".join([str(x) for x in num_samples]),
        "_".join([str(x) for x in layer_size]),
        dropout,
        learning_rate,
    )
    model.save("cora_example_model" + save_str + ".h5")

    # We must also save the target encoding to convert model predictions
    with open("cora_example_encoding" + save_str + ".pkl", "wb") as f:
        pickle.dump([target_encoding], f)
        
    return "cora_example_model" + save_str + ".h5"

# Load the dataset - this assumes it is the CORA dataset
# Load graph edgelist
if args.location is not None:
    graph_loc = os.path.expanduser(args.location)
else:
    raise ValueError(
        "Please specify the directory containing the dataset using the '-l' flag"
    )

edgelist = pd.read_table(
    os.path.join(graph_loc, "cora.cites"), header=None, names=["source", "target"]
)

# Load node features
# The CORA dataset contains binary attributes 'w_x' that correspond to whether the corresponding keyword
# (out of 1433 keywords) is found in the corresponding publication.
feature_names = ["w_{}".format(ii) for ii in range(1433)]
# Also, there is a "subject" column
column_names = feature_names + ["subject"]
node_data = pd.read_table(
    os.path.join(graph_loc, "cora.content"), header=None, names=column_names
)

args.checkpoint = train(
    edgelist,
    node_data,
    args.layer_size,
    args.neighbour_samples,
    args.batch_size,
    args.epochs,
    args.learningrate,
    args.dropout,
)

########### Get Pred Labels (Right now we do not have seperate testing and training..all in sample to demonstrate visual)

# Extract the feature data. These are the feature vectors that the Keras model will use as input.
# The CORA dataset contains attributes 'w_x' that correspond to words found in that publication.
node_features = node_data[feature_names]
node_ids = node_data.index
Gnx = nx.from_pandas_edgelist(edgelist)

encoder_file = args.checkpoint.replace(
    "cora_example_model", "cora_example_encoding"
).replace(".h5", ".pkl")
with open(encoder_file, "rb") as f:
    target_encoding = pickle.load(f)[0]

# Endode targets with pre-trained encoder
node_targets = target_encoding.transform(
    node_data[[args.target]].to_dict("records")
)

# Convert to StellarGraph and prepare for ML
G = sg.StellarGraph(Gnx, node_features=node_features)
    
model = keras.models.load_model(
    args.checkpoint, custom_objects={"MeanAggregator": MeanAggregator}
)
num_samples = [
    int(model.input_shape[ii + 1][1] / model.input_shape[ii][1])
    for ii in range(len(model.input_shape) - 1)
]
# Create mappers for GraphSAGE that input data from the graph to the model
generator = GraphSAGENodeGenerator(G, args.batch_size, num_samples, seed=42)
all_gen = generator.flow(node_ids, node_targets)

# Get predictions for all nodes
all_predictions = model.predict_generator(all_gen)

# Turn predictions back into the original categories
node_predictions = pd.DataFrame(
    target_encoding.inverse_transform(all_predictions), index=node_ids
)

# concat predictions back to the data
node_predictions['pred_subject'] = node_predictions.idxmax(axis=1).str.replace('subject=','')
node_data = node_data.join(node_predictions, how = 'inner')
sum(node_data.subject == node_data.pred_subject)/node_data.shape[0]

##### Calculate Centrality Metrics
Gnx_d = nx.from_pandas_edgelist(edgelist, create_using=nx.DiGraph())
degree_centrality = pd.Series(nx.algorithms.centrality.degree_centrality(Gnx_d))
in_degree_centrality = pd.Series(nx.algorithms.centrality.in_degree_centrality(Gnx_d))
out_degree_centrality = pd.Series(nx.algorithms.centrality.out_degree_centrality(Gnx_d))
centrality_metrics = pd.concat([degree_centrality,in_degree_centrality,out_degree_centrality],axis = 1)
centrality_metrics.columns = ['degree', 'in_degree','out_degree']
node_data = node_data.join(centrality_metrics, how = 'inner')


#bb = nx.readwrite.json_graph.cytoscape_data(Gnx_d) #We will do a manual conversion  rather than using the built in method to get the format we want 

######## Make Cytoscape elements

node_schema = {
   'data':{          
        'id' : 0,
        'label' : 'INIT',
        'actual_subject' : 'INIT',
        'pred_subject' : 'INIT',
        'w_indicators' : [],
        'pc_degree' : -1.0, #pre calculated degree centrality
        'pc_in_degree' : -1.0, #pre calculated in degree centrality
        'pc_out_degree' : -1.0 #pre calculated out degree centrality   
   }

}
edge_schema = {
   'data':{          
      'id' : 0,
      'source' : 'INIT',
      'target' : 'INIT'    
   }  
}

nodes = []
edges = []

for index, row in node_data.iterrows():
    node = {'data':{}}
    node['data']['id'] = str(index)
    node['data']['label'] = str(index)
    node['data']['w_indicators'] = row[row == 1].index.tolist()
    node['data']['actual_subject'] = row.subject
    node['data']['pred_subject'] = row.pred_subject
    node['data']['pc_degree'] = row.degree
    node['data']['pc_in_degree'] = row.in_degree
    node['data']['pc_out_degree'] = row.out_degree
    nodes.append(node.copy())
    
for index, row in edgelist.iterrows():
    edge = {'data':{}}
    s = str(row.source)
    t = str(row.target)
    edge['data']['id'] = s + '-to-' + t
    edge['data']['source'] = s
    edge['data']['target'] = t
    edges.append(edge.copy())
 

##This is big.  Sample 500 edges only
edges_sample = random.sample(population=edges, k=300)
  
sample_nodes_ids = []
nodes_sample = []
for item in edges_sample:
     n_s = item['data']['source']
     if n_s not in sample_nodes_ids:
        sample_nodes_ids.append(n_s)
        node = {'data':{}}
        row = node_data.loc[int(n_s),:]
        node['data']['id'] = n_s
        node['data']['label'] = n_s
        node['data']['w_indicators'] = row[row == 1].index.tolist()
        node['data']['actual_subject'] = row.subject
        node['data']['pred_subject'] = row.pred_subject
        node['data']['pc_degree'] = row.degree
        node['data']['pc_in_degree'] = row.in_degree
        node['data']['pc_out_degree'] = row.out_degree
        nodes_sample.append(node.copy())
        
     n_t = item['data']['target']
     if n_t not in sample_nodes_ids:
        sample_nodes_ids.append(n_t)
        node = {'data':{}}
        row = node_data.loc[int(n_t),:]
        node['data']['id'] = n_t
        node['data']['label'] = n_t
        node['data']['w_indicators'] = row[row == 1].index.tolist()
        node['data']['actual_subject'] = row.subject
        node['data']['pred_subject'] = row.pred_subject
        node['data']['pc_degree'] = row.degree
        node['data']['pc_in_degree'] = row.in_degree
        node['data']['pc_out_degree'] = row.out_degree    
        nodes_sample.append(node.copy())
        
        
graphP = {'elements':{'nodes':nodes_sample,'edges':edges_sample}}
temp_string = json.dumps(graphP)
filename = args.cy_json_outfile + '_' + hashlib.md5(temp_string.encode('utf-8')).hexdigest() +'.json'
with open(filename , 'w') as wf:
    json.dump(graphP, wf)