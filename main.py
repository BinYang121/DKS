import json
import scipy.sparse as sp
from utils import *
from dan_net import *
from gat_utils import *
from gat_trainer import *
import torch_geometric.data as data

# Para
dataset = 'data/citeseer/'
graph_path = 'contaminated_graph/'  # 'contaminated_graph/'   'incomplete_graph/'
edge_file = 'edge_index.npz'  # 'edge_index.npz'  'incomplete_graph/edge_30_edge_index.npz'
spx_file = '30%contaminate_spX.npz'
clean_spx_file = 'spX.npz'
dirty_nodes_file = '30%contaminate_nodes.out'
clean_sample_num = 5
clean_sample_k_hop_ratio = 0.6
clean_sample_ratio = 0.6
clean_train_k_hop = 3
decay_theta = 0.5
k_hop = 7

# test_file path generate
test_files, tops_k_record = [], []
kws = [3, 5, 7, 9]
tops_k = [10, 50, 100]
test_path = 'test/'
for i in range(len(kws)):
    for j in range(len(tops_k)):
        kw_dir = 'kw' + str(kws[i])
        test_file = test_path + kw_dir + '/' + kw_dir + '_top' + str(tops_k[j]) + '_output.txt'
        test_files.append(test_file)
        tops_k_record.append(tops_k[j])
print("Finish generate the test file path!")

# Read inputs
start_time = time.time()
spX = sp.load_npz(dataset + graph_path + spx_file)
v_kw_matrix = spX.toarray().tolist()
clean_spX = sp.load_npz(dataset + clean_spx_file)
clean_v_kw_matrix = clean_spX.toarray().tolist()

edge_index = np.load(dataset + edge_file)
edges = edge_index['arr_0']
edges_list = edges.tolist()
edge_index = torch.LongTensor([edges_list[0] + edges_list[1], edges_list[1] + edges_list[0]])
g = data.Data(edge_index=edge_index, num_nodes=len(v_kw_matrix))
edges_dict = {}
edges_dict_tmp = [[] for _ in range(len(v_kw_matrix))]
for i in range(len(edges[0])):
    edges_dict_tmp[edges[0][i]].append(edges[1][i])
    edges_dict_tmp[edges[1][i]].append(edges[0][i])
for i in range(len(v_kw_matrix)):
    edges_dict.update({i: sorted(list(set(edges_dict_tmp[i])))})

dirty_nodes_gt = read_workload(dataset + graph_path, dirty_nodes_file)
print("Finish read dirty_nodes_gt.")

# Dirty nodes find & Sample clean nodes for cleaning
dirty_nodes = []
if graph_path == 'contaminated_graph/':
    print("The type of the graph: Contaminated graph.")
    print(graph_path + spx_file)
    dirty_nodes = dirty_check_2(v_kw_matrix, clean_train_k_hop, g, dirty_nodes_gt)
    print("PCA finds the number of contaminated nodes:", len(dirty_nodes))
elif graph_path == 'incomplete_graph/':
    print("The type of the graph: Incomplete graph.")
    print(graph_path + spx_file)
    print(edge_file)
    dirty_nodes = dirty_check_3(v_kw_matrix, dirty_nodes_gt)
    print("Finding the number of incomplete nodes:", len(dirty_nodes))
else:
    print("Wrong: no this dirty mode!")
clean_train_data = sample_clean_train(v_kw_matrix, g, dirty_nodes, clean_sample_num, clean_sample_k_hop_ratio,
                                      clean_train_k_hop, clean_sample_ratio)
clean_test_data, clean_exp_test_data = sample_clean_test(v_kw_matrix, g, dirty_nodes, clean_sample_num,
                                                         clean_sample_k_hop_ratio, clean_train_k_hop, clean_v_kw_matrix)

# Gat params
with open("gat_params.json") as f:
    gat_config = json.load(f)
gat_device = gpu_setup(gat_config['gpu']['use'], gat_config['gpu']['id'])
gat_params = gat_config['params']
gat_net_params = gat_config['net_params']
gat_net_params['device'] = gat_device
gat_net_params['gpu_id'] = gat_config['gpu']['id']
gat_net_params['input_dim'] = len(v_kw_matrix[0])
gat_batch_size = gat_config['params']['batch_size']

# Gat
random.shuffle(clean_train_data)
train_gat_matrices, train_gat_graphs, train_gat_targets = data_format(clean_train_data)
random.shuffle(clean_test_data)
test_gat_matrices, test_gat_graphs, test_gat_targets = data_format(clean_test_data)
train_batch_loaders, train_batch_matrices, train_batch_targets = gat_batch(train_gat_matrices, train_gat_graphs,
                                                                           train_gat_targets, gat_batch_size)
test_batch_loaders, test_batch_matrices, test_batch_targets = gat_batch(test_gat_matrices, test_gat_graphs,
                                                                        test_gat_targets, gat_batch_size)
print("Train data:", len(clean_train_data), "Test data:", len(clean_test_data))
exp_test_batch_data = exp_data_batch(clean_exp_test_data, clean_sample_num)
print("The number of cleaned nodes:", len(exp_test_batch_data))
data = [[train_batch_loaders, train_batch_matrices, train_batch_targets],
        [test_batch_loaders, test_batch_matrices, test_batch_targets]]
clean_nodes_vec = train_pipeline(gat_params, gat_net_params, data, exp_test_batch_data)

# Generate the cleaned v_kw_matrix
cleaned_v_kw_matrix = []
for i in range(len(v_kw_matrix)):
    if i in clean_nodes_vec:
        cleaned_v_kw_matrix.append(clean_nodes_vec[i])
    else:
        cleaned_v_kw_matrix.append(v_kw_matrix[i])
print("Finish clean the data graph.")

# Compare the cleaned cleaned_v_kw_matrix & the clean clean_v_kw_matrix
xor = [[int(a) ^ int(b) for a, b in zip(row1, row2)] for row1, row2 in zip(cleaned_v_kw_matrix, clean_v_kw_matrix)]
xor_sum = 0
for row in xor:
    for num in row:
        xor_sum += num
error_ratio = xor_sum / (len(clean_v_kw_matrix) * len(clean_v_kw_matrix[0]))
print("The error ratio between cleaned_v_kw_matrix and clean_v_kw_matrix:", error_ratio)
print("*" * 89)
print("*" * 89)

# Execute query for each test file
for t in range(len(test_files)):
    test_file = test_files[t]
    top_k = tops_k_record[t]
    print("Test file: ", test_file)

    # Read test file and One-hot transform for queries
    test_workload = read_queries(dataset, test_file)
    print("Finish read test_file.")
    one_hot_test = []
    for i in range(len(test_workload['qs'])):
        zeros = [0] * len(v_kw_matrix[0])
        for j in test_workload['qs'][i]:
            zeros[j] = 1
        one_hot_test.append(zeros)

    # GNN + DAN (i.e., DKS):
    nodes_num = len(cleaned_v_kw_matrix)
    print("GNN + DAN (i.e., DKS):")
    tmp_matrix_dict_d = {}
    cleaned_k_hop_matrices, tmp_matrix_dict_d = k_aggregate(k_hop, cleaned_v_kw_matrix, edges_dict, decay_theta,
                                                            tmp_matrix_dict_d)
    one_hot_test_tensor = torch.tensor(one_hot_test).long()
    cleaned_k_hop_matrix_tensor = torch.tensor(cleaned_k_hop_matrices).long()
    scores = torch.matmul(one_hot_test_tensor, cleaned_k_hop_matrix_tensor.t())
    rank = scores.sort(dim=-1, descending=True)[1]
    hit_valid = eval_Z(nodes_num, rank, test_workload['ans'], top_k)
    print("The number of layers:", k_hop, " Accuracy:", hit_valid)
    print('-' * 89)
    print('-' * 89)

end_time = time.time()
print("The total time:", end_time - start_time)
