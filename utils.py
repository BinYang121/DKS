import torch
import random
import numpy as np
import networkx as nx
from sklearn.decomposition import PCA
from torch_geometric.utils import k_hop_subgraph, to_networkx


def read_command(f):
    line = f.readline()
    if line != "":
        data = line.strip('\n').split('/n')
        sub_str = data[0].split(' ')
    else:
        return "finish"
    return sub_str


def read_workload(path, file):
    dirty_nodes_gt = []
    input_file = open(path + file, "r")
    sub_str = read_command(input_file)
    while sub_str != "finish":
        dirty_nodes_gt.append(int(sub_str[0]))
        sub_str = read_command(input_file)

    return dirty_nodes_gt


def read_queries(path, query_file):
    query_data = {}
    with open(path + query_file, 'r') as test_f:
        for line in test_f.readlines():
            query, ans = line.strip().split('\t')
            query_data[query] = list(map(int, ans.split()))

    query_qs, query_ans = [], []
    for q in query_data.keys():
        query_qs.append(list(map(int, q.strip().split())))
        query_ans.append(query_data[q])

    query_workload = {'qs': query_qs, 'ans': query_ans}

    return query_workload


def dirty_check_1(v_kw_matrix, edges_dict, dirty_nodes_gt):
    ratios, correct_dirty, predict_dirty = [], [], []
    nodes = list(range(0, len(v_kw_matrix)))
    dirty_nodes_gt = set(dirty_nodes_gt)
    for i in range(len(v_kw_matrix)):
        one_hop_kw = [v_kw_matrix[i]]
        hop_nodes = [i]
        for neighbor in edges_dict[i]:
            one_hop_kw.append(v_kw_matrix[neighbor])
            hop_nodes.append(neighbor)
        if len(one_hop_kw) > 1:
            pca = PCA(n_components=1)
            pca.fit(one_hop_kw)
            reduced_data = pca.transform(one_hop_kw)
            deviation_threshold = 1
            mean_value = np.mean(reduced_data, axis=0)
            std_deviation = np.std(reduced_data, axis=0)
            outlier_indices = [i for i, v in enumerate(reduced_data) if
                               abs(v[0] - mean_value[0]) > deviation_threshold * std_deviation[0]]
            dirty_nodes = [hop_nodes[node] for node in outlier_indices]
            if len(dirty_nodes) != 0:
                # correct_dirty += list(set(dirty_nodes).intersection(dirty_nodes_gt))
                predict_dirty += dirty_nodes
                # ratios.append(len(set(dirty_nodes).intersection(dirty_nodes_gt)) / len(dirty_nodes))
    tp = set(predict_dirty).intersection(dirty_nodes_gt)
    tn = set(set(nodes) - dirty_nodes_gt).intersection(set(nodes) - set(predict_dirty))
    fp = set(predict_dirty) - dirty_nodes_gt
    fn = (set(nodes) - set(predict_dirty)).intersection(dirty_nodes_gt)
    print("TP:", len(tp), "TN:", len(tn), "FP", len(fp), "FN", len(fn))
    print("TP+FN:", len(dirty_nodes_gt))
    print("FP+TN:", len(set(nodes) - dirty_nodes_gt))
    print("TP+FP:", len(set(predict_dirty)))
    precision = len(tp) / (len(tp) + len(fp))
    recall = len(tp) / (len(tp) + len(fn))
    f1 = 2 * precision * recall / (precision + recall)
    print("Precision:", precision, "Recall:", recall, "F1:", f1)


def dirty_check_2(v_kw_matrix, k_hop, graph, dirty_nodes_gt):
    ratios, correct_dirty, predict_dirty = [], [], []
    nodes = list(range(0, len(v_kw_matrix)))
    dirty_nodes_gt = set(dirty_nodes_gt)
    for i in range(len(v_kw_matrix)):
        one_hop_kw = [v_kw_matrix[i]]
        hop_nodes = [i]
        sub_graph = k_hop_subgraph(node_idx=[i], num_hops=k_hop, edge_index=graph.edge_index)
        org_nodes = sub_graph[0].tolist()
        for node in org_nodes:
            one_hop_kw.append(v_kw_matrix[node])
            hop_nodes.append(node)
        if len(one_hop_kw) > 3:
            pca = PCA(n_components=1)
            pca.fit(one_hop_kw)
            reduced_data = pca.transform(one_hop_kw)
            deviation_threshold = 1
            mean_value = np.mean(reduced_data, axis=0)
            std_deviation = np.std(reduced_data, axis=0)
            outlier_indices = [i for i, v in enumerate(reduced_data) if
                               abs(v[0] - mean_value[0]) > deviation_threshold * std_deviation[0]]
            dirty_nodes = [hop_nodes[node] for node in outlier_indices]
            if len(dirty_nodes) != 0:
                # correct_dirty += list(set(dirty_nodes).intersection(dirty_nodes_gt))
                predict_dirty += dirty_nodes
                # ratios.append(len(set(dirty_nodes).intersection(dirty_nodes_gt)) / len(dirty_nodes))
    tp = set(predict_dirty).intersection(dirty_nodes_gt)
    tn = set(set(nodes) - dirty_nodes_gt).intersection(set(nodes) - set(predict_dirty))
    fp = set(predict_dirty) - dirty_nodes_gt
    fn = (set(nodes) - set(predict_dirty)).intersection(dirty_nodes_gt)
    print("TP:", len(tp), "TN:", len(tn), "FP", len(fp), "FN", len(fn))
    print("TP+FN:", len(dirty_nodes_gt))
    print("FP+TN:", len(set(nodes) - dirty_nodes_gt))
    print("TP+FP:", len(set(predict_dirty)))
    precision = len(tp) / (len(tp) + len(fp))
    recall = len(tp) / (len(tp) + len(fn))
    f1 = 2 * precision * recall / (precision + recall)
    print("Precision:", precision, "Recall:", recall, "F1:", f1)

    return list(set(predict_dirty))


def dirty_check_3(v_kw_matrix, dirty_nodes_gt):
    predict_dirty = []
    for i in range(len(v_kw_matrix)):
        if sum(v_kw_matrix[i]) == 0:
            predict_dirty.append(i)
    nodes = list(range(0, len(v_kw_matrix)))
    dirty_nodes_gt = set(dirty_nodes_gt)
    tp = set(predict_dirty).intersection(dirty_nodes_gt)
    tn = set(set(nodes) - dirty_nodes_gt).intersection(set(nodes) - set(predict_dirty))
    fp = set(predict_dirty) - dirty_nodes_gt
    fn = (set(nodes) - set(predict_dirty)).intersection(dirty_nodes_gt)
    print("TP:", len(tp), "TN:", len(tn), "FP", len(fp), "FN", len(fn))
    print("TP+FN:", len(dirty_nodes_gt))
    print("FP+TN:", len(set(nodes) - dirty_nodes_gt))
    print("TP+FP:", len(set(predict_dirty)))
    precision = len(tp) / (len(tp) + len(fp))
    recall = len(tp) / (len(tp) + len(fn))
    f1 = 2 * precision * recall / (precision + recall)
    print("Precision:", precision, "Recall:", recall, "F1:", f1)

    return predict_dirty


def sample_clean_train(v_kw_matrix, graph, dirty_nodes, sample_num, sample_k_hop_ratio, k_hop, sample_ratio):
    nx_graph = to_networkx(graph)
    train_data = []
    train_id = 0
    nodes = [i for i in range(len(v_kw_matrix))]
    clean_nodes = [item for item in nodes if item not in dirty_nodes]
    sample_clean_nodes = random.sample(clean_nodes, int(len(clean_nodes) * sample_ratio))
    print('The number of sample_clean_nodes: ', len(sample_clean_nodes))
    num = 0
    for i in sample_clean_nodes:
        num += 1
        sub_graph = k_hop_subgraph(node_idx=[i], num_hops=k_hop, edge_index=graph.edge_index)
        org_nodes = sub_graph[0].tolist()
        clean_nodes = [item for item in org_nodes if item not in dirty_nodes]
        for j in range(sample_num):
            sample_nodes = random.sample(clean_nodes, int(len(clean_nodes) * sample_k_hop_ratio))
            if i not in sample_nodes:
                sample_nodes = sample_nodes + [i]
            if len(sample_nodes) > 1:
                record_spl = [[0 for _ in range(len(sample_nodes))] for _ in range(len(sample_nodes))]
                for node_i in range(len(sample_nodes)):
                    for node_j in range(len(sample_nodes)):
                        if record_spl[node_i][node_j] == 0 and node_i != node_j:
                            dis = nx.shortest_path_length(nx_graph, sample_nodes[node_i], sample_nodes[node_j])
                            record_spl[node_i][node_j] = dis
                            record_spl[node_j][node_i] = dis
                adj = np.zeros((len(sample_nodes), len(sample_nodes)), dtype=int)
                for p in range(len(sample_nodes)):
                    min_indices = [q for q, x in enumerate(record_spl[p]) if
                                   x == min(y for y in record_spl[p] if y > 0)]
                    for r in min_indices:
                        adj[p][r] = 1
                        adj[r][p] = 1
                nodes_kw_matrix = []
                for p in range(len(sample_nodes)):
                    nodes_kw_matrix.append(v_kw_matrix[sample_nodes[p]])
                nodes = [x for x in range(len(sample_nodes))]
                adj_tensor = torch.tensor(adj, dtype=torch.int8)
                target_tensor = torch.tensor(v_kw_matrix[i])
                tt = torch.tensor(float(0))
                train_graph = {'id': train_id, 'data_node_id': i, 'num_v': len(sample_nodes), 'nodes': nodes,
                               'node_labels': nodes_kw_matrix, 'adj': adj, 'adj_tensor': adj_tensor,
                               'target': v_kw_matrix[i], 'target_tensor': target_tensor, 'tt': tt}
                train_data.append(train_graph)
                train_id += 1

    return train_data


def sample_clean_test(v_kw_matrix, graph, dirty_nodes, sample_num, sample_k_hop_ratio, k_hop, clean_v_kw_matrix):
    nx_graph = to_networkx(graph)
    test_data, exp_test_data = [], []
    test_id = 0
    for i in range(len(v_kw_matrix)):
        if i in dirty_nodes:
            sub_graph = k_hop_subgraph(node_idx=[i], num_hops=k_hop, edge_index=graph.edge_index)
            org_nodes = sub_graph[0].tolist()
            clean_nodes = [item for item in org_nodes if item not in dirty_nodes]
            sample_graphs = []
            for j in range(sample_num):
                sample_nodes = random.sample(clean_nodes, int(len(clean_nodes) * sample_k_hop_ratio))
                if i not in sample_nodes:
                    sample_nodes = sample_nodes + [i]
                if len(sample_nodes) > 1:
                    record_spl = [[0 for _ in range(len(sample_nodes))] for _ in range(len(sample_nodes))]
                    for node_i in range(len(sample_nodes)):
                        for node_j in range(len(sample_nodes)):
                            if record_spl[node_i][node_j] == 0 and node_i != node_j:
                                dis = nx.shortest_path_length(nx_graph, sample_nodes[node_i], sample_nodes[node_j])
                                record_spl[node_i][node_j] = dis
                                record_spl[node_j][node_i] = dis
                    adj = np.zeros((len(sample_nodes), len(sample_nodes)), dtype=int)
                    for p in range(len(sample_nodes)):
                        min_indices = [q for q, x in enumerate(record_spl[p]) if
                                       x == min(y for y in record_spl[p] if y > 0)]
                        for r in min_indices:
                            adj[p][r] = 1
                            adj[r][p] = 1
                    nodes_kw_matrix = []
                    for p in range(len(sample_nodes)):
                        nodes_kw_matrix.append(v_kw_matrix[sample_nodes[p]])
                    nodes = [x for x in range(len(sample_nodes))]
                    adj_tensor = torch.tensor(adj, dtype=torch.int8)
                    target_tensor = torch.tensor(clean_v_kw_matrix[i])
                    tt = torch.tensor(float(0))
                    test_graph = {'id': test_id, 'data_node_id': i, 'num_v': len(sample_nodes), 'nodes': nodes,
                                  'node_labels': nodes_kw_matrix, 'adj': adj, 'adj_tensor': adj_tensor,
                                  'target': clean_v_kw_matrix[i], 'target_tensor': target_tensor, 'tt': tt}
                    sample_graphs.append(test_graph)
                    test_data.append(test_graph)
                    test_id += 1
            exp_test_data.append(sample_graphs)

    return test_data, exp_test_data


def eval_Z(nodes_num, rank, ans, k):
    hits = []
    for i in range(len(rank)):
        mark = torch.zeros(nodes_num)
        mark[ans[i]] = 1
        tmp_hit = mark[rank[i, :k]].sum() / k
        hits.append(tmp_hit)
    hits = torch.stack(hits)

    return hits.mean().item()









