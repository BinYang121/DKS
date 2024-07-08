

def aggregate(matrix, edges_dict, theta):
    next_matrix = []
    for i in range(len(matrix)):
        tmp = matrix[i]
        for neighbor in edges_dict[i]:
            tmp = [x + y * theta for x, y in zip(tmp, matrix[neighbor])]
        next_matrix.append(tmp)

    return next_matrix


def k_aggregate(k, matrix, edges_dict, theta, tmp_matrix_dict):
    k_i = 0
    for i in range(k):
        if (k - i) in tmp_matrix_dict:
            k_i = k - i
            break
    if k_i != 0:
        hidden_matrices = tmp_matrix_dict[k_i][0]
        next_matrix = hidden_matrices[-1]
        k_hop = list(range(0, k-k_i))
        for h in k_hop:
            next_matrix = aggregate(next_matrix, edges_dict, theta)
            hidden_matrices.append(next_matrix)
        tmp_hidden_matrix = tmp_matrix_dict[k_i][1]
        for i in range(0, k-k_i):
            tmp_hidden_matrix = [[x + y for x, y in zip(a_row, b_row)] for a_row, b_row in
                                 zip(tmp_hidden_matrix, hidden_matrices[k_i + i + 1])]
    else:
        hidden_matrices = [matrix]
        next_matrix = matrix
        k_hop = list(range(0, k))
        for h in k_hop:
            next_matrix = aggregate(next_matrix, edges_dict, theta)
            hidden_matrices.append(next_matrix)
        tmp_hidden_matrix = hidden_matrices[0]
        for i in range(len(hidden_matrices) - 1):
            tmp_hidden_matrix = [[x + y for x, y in zip(a_row, b_row)] for a_row, b_row in
                                 zip(tmp_hidden_matrix, hidden_matrices[i + 1])]

    tmp_matrix_dict.update({k: [hidden_matrices, tmp_hidden_matrix]})

    return tmp_hidden_matrix, tmp_matrix_dict










