import torch
from torch.nn import Linear, ReLU, Dropout, LSTM
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool as gep, PairNorm


vector_operations = {
    "cat": (lambda x, y: torch.cat((x, y), -1), lambda dim: 2 * dim),
    "add": (torch.add, lambda dim: dim),
    "sub": (torch.sub, lambda dim: dim),
    "mul": (torch.mul, lambda dim: dim),
    "combination1": (lambda x, y: torch.cat((x, y, torch.add(x, y)), -1), lambda dim: 3 * dim),
    "combination2": (lambda x, y: torch.cat((x, y, torch.sub(x, y)), -1), lambda dim: 3 * dim),
    "combination3": (lambda x, y: torch.cat((x, y, torch.mul(x, y)), -1), lambda dim: 3 * dim),
    "combination4": (lambda x, y: torch.cat((torch.add(x, y), torch.sub(x, y)), -1), lambda dim: 2 * dim),
    "combination5": (lambda x, y: torch.cat((torch.add(x, y), torch.mul(x, y)), -1), lambda dim: 2 * dim),
    "combination6": (lambda x, y: torch.cat((torch.sub(x, y), torch.mul(x, y)), -1), lambda dim: 2 * dim),
    "combination7": (
    lambda x, y: torch.cat((torch.add(x, y), torch.sub(x, y), torch.mul(x, y)), -1), lambda dim: 3 * dim),
    "combination8": (lambda x, y: torch.cat((x, y, torch.sub(x, y), torch.mul(x, y)), -1), lambda dim: 4 * dim),
    "combination9": (lambda x, y: torch.cat((x, y, torch.add(x, y), torch.mul(x, y)), -1), lambda dim: 4 * dim),
    "combination10": (lambda x, y: torch.cat((x, y, torch.add(x, y), torch.sub(x, y)), -1), lambda dim: 4 * dim),
    "combination11": (
    lambda x, y: torch.cat((x, y, torch.add(x, y), torch.sub(x, y), torch.mul(x, y)), -1), lambda dim: 5 * dim)
}


class LinearBlock(torch.nn.Module):
    def __init__(self, linear_layers_dim, dropout_rate=0.0, relu_layers_index=[], dropout_layers_index=[]):
        super(LinearBlock, self).__init__()

        self.layers = torch.nn.ModuleList()
        for i in range(len(linear_layers_dim) - 1):
            layer = Linear(linear_layers_dim[i], linear_layers_dim[i + 1])
            self.layers.append(layer)

        self.relu = ReLU()
        self.dropout = Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x):
        output = x
        embeddings = [x]
        for layer_index in range(len(self.layers)):
            output = self.layers[layer_index](output)
            if layer_index in self.relu_layers_index:
                output = self.relu(output)
            if layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            embeddings.append(output)
        return embeddings



class GCNBlock(torch.nn.Module):
    def __init__(self, gcn_layers_dim, dropout_rate=0, relu_layers_index=[], dropout_layers_index=[],
                 supplement_mode=None):
        super(GCNBlock, self).__init__()

        self.conv_layers = torch.nn.ModuleList()
        for i in range(len(gcn_layers_dim) - 1):
            if supplement_mode is not None and i == 1:
                conv_layer_input = gcn_layers_dim[i] * 3
            else:
                conv_layer_input = gcn_layers_dim[i]
            conv_layer = GCNConv(conv_layer_input, gcn_layers_dim[i + 1])
            self.conv_layers.append(conv_layer)

        self.relu = ReLU()
        self.dropout = Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x, edge_index, edge_weight, batch, supplement_x=None):
        output = x
        embeddings = [x]

        for conv_layer_index in range(len(self.conv_layers)):
            if supplement_x is not None and conv_layer_index == 1:
                y = supplement_x
                output = torch.cat((torch.add(output, y), torch.sub(output, y), output), -1)

            output = self.conv_layers[conv_layer_index](output, edge_index, edge_weight)
            if conv_layer_index in self.relu_layers_index:
                output = self.relu(output)
            if conv_layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            embeddings.append(gep(output, batch))
        return embeddings

class GCNModel(torch.nn.Module):
    def __init__(self, layers_dim, supplement_mode=None):
        super(GCNModel, self).__init__()

        self.num_layers = len(layers_dim) - 1
        self.graph_conv = GCNBlock(layers_dim, relu_layers_index=range(self.num_layers),
                                   supplement_mode=supplement_mode)

    def forward(self, graph_batchs, supplement_x=None):

        if supplement_x is not None:
            supplement_i = 0
            for graph_batch in graph_batchs:
                graph_batch.__setitem__('supplement_x',
                                        supplement_x[supplement_i: supplement_i + graph_batch.num_graphs])
                supplement_i += graph_batch.num_graphs

            embedding_batchs = list(map(lambda graph: self.graph_conv(graph.x, graph.edge_index, None, graph.batch,
                                                                      supplement_x=graph.supplement_x[
                                                                          graph.batch.int().cpu().numpy()]),
                                        graph_batchs))
        else:
            embedding_batchs = list(
                map(lambda graph: self.graph_conv(graph.x, graph.edge_index, None, graph.batch), graph_batchs))

        embeddings = []
        for i in range(self.num_layers + 1):
            embeddings.append(torch.cat(list(map(lambda embedding_batch: embedding_batch[i], embedding_batchs)), 0))

        return embeddings

class GCNlayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2, train_eps=True, residual=True):
        super(GCNlayer, self).__init__()
        self.residual = residual

        # GCNConv
        self.gcn_conv = GCNConv(in_channels, out_channels)

        self.norm = PairNorm()

        self.dropout = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, x, edge_index, edge_weight = None):

        if edge_weight == None:
            out = self.gcn_conv(x, edge_index)
        else:
            out = self.gcn_conv(x, edge_index, edge_weight)

        out = self.norm(out)
        out = self.activation(out)

        # Dropout
        out = self.dropout(out)

        return out

class MGEL(torch.nn.Module):
    def __init__(self, mg_init_dim=78, pg_init_dim=54,  embedding_dim=128):
        super(MGEL, self).__init__()

        drug_graph_dims = [mg_init_dim, mg_init_dim, mg_init_dim * 3, mg_init_dim * 4]
        target_graph_dims = [pg_init_dim, pg_init_dim, pg_init_dim * 3, pg_init_dim * 4]

        self.drug_graph_conv = GCNModel(drug_graph_dims, supplement_mode='combination4')
        self.target_graph_conv = GCNModel(target_graph_dims,  supplement_mode='combination4')

        drug_similarity_dims = [384, 512, 512, 256]
        target_similarity_dims = [768, 512, 512, 256]

        self.drug_1 = GCNlayer(drug_similarity_dims[0], drug_similarity_dims[1])
        self.drug_2 = GCNlayer(drug_similarity_dims[1], drug_similarity_dims[2])
        self.drug_3 = GCNlayer(drug_similarity_dims[2], drug_similarity_dims[3])
        
        self.target_1 =  GCNlayer(target_similarity_dims[0], target_similarity_dims[1])
        self.target_2 =  GCNlayer(target_similarity_dims[1], target_similarity_dims[2])
        self.target_3 =  GCNlayer(target_similarity_dims[2], target_similarity_dims[3])

        drug_transform_dims = [256, 1024, drug_graph_dims[1]]
        target_transform_dims = [256, 1024, target_graph_dims[1]]

        self.drug_trans = LinearBlock(drug_transform_dims, 0.2, [0],[0])
        self.target_trans = LinearBlock(target_transform_dims, 0.2, [0], [0])


        drug_aggr_dims = [drug_graph_dims[-1] + 256, 1024, embedding_dim]
        drug_s_dims = [256, 1024, embedding_dim]
        drug_g_dims = [drug_graph_dims[-1], 1024, embedding_dim]

        self.drug_aggr_linear = LinearBlock(drug_aggr_dims, 0.2, relu_layers_index=[0], dropout_layers_index=[0, 1])
        self.drug_s_linear = LinearBlock(drug_s_dims, 0.2, relu_layers_index=[0], dropout_layers_index=[0, 1])
        self.drug_g_linear = LinearBlock(drug_g_dims, 0.2, relu_layers_index=[0], dropout_layers_index=[0, 1])

        target_aggr_dims = [target_graph_dims[-1] + 256, 1024, embedding_dim]
        target_s_dims = [256, 1024, embedding_dim]
        target_g_dims = [target_graph_dims[-1], 1024, embedding_dim]

        self.target_aggr_linear = LinearBlock(target_aggr_dims, 0.2, relu_layers_index=[0], dropout_layers_index=[0, 1])
        self.target_s_linear = LinearBlock(target_s_dims, 0.2, relu_layers_index=[0], dropout_layers_index=[0, 1])
        self.target_g_linear = LinearBlock(target_g_dims, 0.2, relu_layers_index=[0], dropout_layers_index=[0, 1])

        
        j_dims = [256, 1024, 512, 1]
        self.j11 = LinearBlock(j_dims, 0.2, [0, 1], [0, 1])
        self.j12 = LinearBlock(j_dims, 0.2, [0, 1], [0, 1])
        self.j13 = LinearBlock(j_dims, 0.2, [0, 1], [0, 1])
        self.j21 = LinearBlock(j_dims, 0.2, [0, 1], [0, 1])
        self.j22 = LinearBlock(j_dims, 0.2, [0, 1], [0, 1])
        self.j23 = LinearBlock(j_dims, 0.2, [0, 1], [0, 1])
        self.j31 = LinearBlock(j_dims, 0.2, [0, 1], [0, 1])
        self.j32 = LinearBlock(j_dims, 0.2, [0, 1], [0, 1])
        self.j33 = LinearBlock(j_dims, 0.2, [0, 1], [0, 1])

        self.mlp = Linear(9,1)


    def forward(self, drug_graph_batchs, target_graph_batchs,  data, drug_similarity_graph, target_similarity_graph):

        drug_x, drug_edge, drug_weight = drug_similarity_graph
        target_x, target_edge, target_weight = target_similarity_graph
        target_x = target_x.float()


        drug_1 = self.drug_1(drug_x, drug_edge, drug_weight)
        drug_2 = self.drug_2(drug_1, drug_edge, drug_weight)
        drug_3 = self.drug_3(drug_2, drug_edge, drug_weight)
        # drug_1 = self.drug_1(drug_x, drug_edge)
        # drug_2 = self.drug_2(drug_1, drug_edge)
        # drug_3 = self.drug_3(drug_2, drug_edge)

        drug_s = self.drug_trans(drug_3)[-1]
        drug_graph_embedding = self.drug_graph_conv(drug_graph_batchs, supplement_x=drug_s)[-1]


        target_1 = self.target_1(target_x, target_edge, target_weight)
        target_2 = self.target_2(target_1, target_edge, target_weight)
        target_3 = self.target_3(target_2, target_edge, target_weight)
        # target_1 = self.target_1(target_x, target_edge)
        # target_2 = self.target_2(target_1, target_edge)
        # target_3 = self.target_3(target_2, target_edge)


        target_s = self.target_trans(target_3)[-1]
        target_graph_embedding = self.target_graph_conv(target_graph_batchs, supplement_x=target_s)[-1]


    
        drug_embedding = torch.cat((drug_3, drug_graph_embedding), 1)
        target_embedding = torch.cat((target_3, target_graph_embedding), 1)


        drug_id, target_id, y = data.drug_id, data.target_id, data.y
        drug_3= drug_3[drug_id.int().cpu().numpy()]
        drug_graph_embedding = drug_graph_embedding[drug_id.int().cpu().numpy()]
        drug_embedding = drug_embedding[drug_id.int().cpu().numpy()]
        target_3 = target_3[target_id.int().cpu().numpy()]
        target_graph_embedding = target_graph_embedding[target_id.int().cpu().numpy()]
        target_embedding = target_embedding[target_id.int().cpu().numpy()]

        d1 = self.drug_s_linear(drug_3)[-1]
        d2 = self.drug_g_linear(drug_graph_embedding)[-1]
        d3 = self.drug_aggr_linear(drug_embedding)[-1]

        t1 = self.target_s_linear(target_3)[-1]
        t2 = self.target_g_linear(target_graph_embedding)[-1]
        t3 = self.target_aggr_linear(target_embedding)[-1]


        e11 = torch.cat((d1, t1), 1)
        e12 = torch.cat((d1, t2), 1)
        e13 = torch.cat((d1, t3), 1)
        e21 = torch.cat((d2, t1), 1)
        e22 = torch.cat((d2, t2), 1)
        e23 = torch.cat((d2, t3), 1)
        e31 = torch.cat((d3, t1), 1)
        e32 = torch.cat((d3, t2), 1)
        e33 = torch.cat((d3, t3), 1)


        ans = self.j11(e11)[-1]
        ans = torch.cat((ans, self.j12(e12)[-1]), 1)
        ans = torch.cat((ans, self.j13(e13)[-1]), 1)
        ans = torch.cat((ans, self.j21(e21)[-1]), 1)
        ans = torch.cat((ans, self.j22(e22)[-1]), 1)
        ans = torch.cat((ans, self.j23(e23)[-1]), 1)
        ans = torch.cat((ans, self.j31(e31)[-1]), 1)
        ans = torch.cat((ans, self.j32(e32)[-1]), 1)
        ans = torch.cat((ans, self.j33(e33)[-1]), 1)

        return self.mlp(ans)
