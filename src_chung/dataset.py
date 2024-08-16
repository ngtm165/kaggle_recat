import numpy as np
import torch
from dgl.convert import graph


class GraphDataset:
    def __init__(self, graph_save_path):
        self.graph_save_path = graph_save_path
        self.load()

    def load(self):
        rmol_dict=np.load(self.graph_save_path,allow_pickle=True)["rmol"]
        pmol_dict=np.load(self.graph_save_path,allow_pickle=True)["pmol"]
        rgmol_dict=np.load(self.graph_save_path,allow_pickle=True)["rgmol"]                   #have just added
        reaction_dict=np.load(self.graph_save_path,allow_pickle=True)["reaction"].item()

        self.rmol_max_cnt = len(rmol_dict)
        self.pmol_max_cnt = len(pmol_dict)
        self.rgmol_max_cnt = len(rgmol_dict)                                                   #have just added

        #reactant
        self.rmol_n_node = [rmol_dict[j]["n_node"] for j in range(self.rmol_max_cnt)]
        self.rmol_n_edge = [rmol_dict[j]["n_edge"] for j in range(self.rmol_max_cnt)]
        self.rmol_node_attr = [
            rmol_dict[j]["node_attr"] for j in range(self.rmol_max_cnt)
        ]
        self.rmol_edge_attr = [
            rmol_dict[j]["edge_attr"] for j in range(self.rmol_max_cnt)
        ]
        self.rmol_src = [rmol_dict[j]["src"] for j in range(self.rmol_max_cnt)]
        self.rmol_dst = [rmol_dict[j]["dst"] for j in range(self.rmol_max_cnt)]

        #product
        self.pmol_n_node = [pmol_dict[j]["n_node"] for j in range(self.pmol_max_cnt)]
        self.pmol_n_edge = [pmol_dict[j]["n_edge"] for j in range(self.pmol_max_cnt)]
        self.pmol_node_attr = [
            pmol_dict[j]["node_attr"] for j in range(self.pmol_max_cnt)
        ]
        self.pmol_edge_attr = [
            pmol_dict[j]["edge_attr"] for j in range(self.pmol_max_cnt)
        ]
        self.pmol_src = [pmol_dict[j]["src"] for j in range(self.pmol_max_cnt)]
        self.pmol_dst = [pmol_dict[j]["dst"] for j in range(self.pmol_max_cnt)]

        #reagent
        self.rgmol_n_node = [rgmol_dict[j]["n_node"] for j in range(self.rgmol_max_cnt)]   #have just added
        self.rgmol_n_edge = [rgmol_dict[j]["n_edge"] for j in range(self.rgmol_max_cnt)]   #have just added
        self.rgmol_node_attr = [rgmol_dict[j]["node_attr"] for j in range(self.rgmol_max_cnt)]   #have just added
        self.rgmol_edge_attr = [rgmol_dict[j]["edge_attr"] for j in range(self.rgmol_max_cnt)]   #have just added
        self.rgmol_src = [rgmol_dict[j]["src"] for j in range(self.rgmol_max_cnt)]   #have just added
        self.rgmol_dst = [rgmol_dict[j]["dst"] for j in range(self.rgmol_max_cnt)]   #have just added


        self.y = reaction_dict["y"]
        self.rsmi = reaction_dict["rsmi"]


        #add csum reactant
        self.rmol_n_csum = [
            np.concatenate([[0], np.cumsum(self.rmol_n_node[j])])
            for j in range(self.rmol_max_cnt)
        ]
        self.rmol_e_csum = [
            np.concatenate([[0], np.cumsum(self.rmol_n_edge[j])])
            for j in range(self.rmol_max_cnt)
        ]

        #add csum product
        self.pmol_n_csum = [
            np.concatenate([[0], np.cumsum(self.pmol_n_node[j])])
            for j in range(self.pmol_max_cnt)
        ]
        self.pmol_e_csum = [
            np.concatenate([[0], np.cumsum(self.pmol_n_edge[j])])
            for j in range(self.pmol_max_cnt)
        ]

        #add csum reagent
        self.rgmol_n_csum = [
            np.concatenate([[0], np.cumsum(self.rgmol_n_node[j])])
            for j in range(self.rgmol_max_cnt)
        ]
        self.rgmol_e_csum = [
            np.concatenate([[0], np.cumsum(self.rgmol_n_edge[j])])
            for j in range(self.rgmol_max_cnt)
        ]

    def __getitem__(self, idx):

        #reactant
        g1 = [
            graph(
                (
                    self.rmol_src[j][
                        self.rmol_e_csum[j][idx] : self.rmol_e_csum[j][idx + 1]
                    ],
                    self.rmol_dst[j][
                        self.rmol_e_csum[j][idx] : self.rmol_e_csum[j][idx + 1]
                    ],
                ),
                num_nodes=self.rmol_n_node[j][idx],
            )
            for j in range(self.rmol_max_cnt)
        ]

        for j in range(self.rmol_max_cnt):
            g1[j].ndata["attr"] = torch.from_numpy(
                self.rmol_node_attr[j][
                    self.rmol_n_csum[j][idx] : self.rmol_n_csum[j][idx + 1]
                ]
            ).float()
            g1[j].edata["edge_attr"] = torch.from_numpy(
                self.rmol_edge_attr[j][
                    self.rmol_e_csum[j][idx] : self.rmol_e_csum[j][idx + 1]
                ]
            ).float()


        #product
        g2 = [
            graph(
                (
                    self.pmol_src[j][
                        self.pmol_e_csum[j][idx] : self.pmol_e_csum[j][idx + 1]
                    ],
                    self.pmol_dst[j][
                        self.pmol_e_csum[j][idx] : self.pmol_e_csum[j][idx + 1]
                    ],
                ),
                num_nodes=self.pmol_n_node[j][idx],
            )
            for j in range(self.pmol_max_cnt)
        ]

        for j in range(self.pmol_max_cnt):
            g2[j].ndata["attr"] = torch.from_numpy(
                self.pmol_node_attr[j][
                    self.pmol_n_csum[j][idx] : self.pmol_n_csum[j][idx + 1]
                ]
            ).float()
            g2[j].edata["edge_attr"] = torch.from_numpy(
                self.pmol_edge_attr[j][
                    self.pmol_e_csum[j][idx] : self.pmol_e_csum[j][idx + 1]
                ]
            ).float()

        
        #reagent
        rg = [
            graph(
                (
                    self.rgmol_src[j][
                        self.rgmol_e_csum[j][idx] : self.rgmol_e_csum[j][idx + 1]
                    ],
                    self.rgmol_dst[j][
                        self.rgmol_e_csum[j][idx] : self.rgmol_e_csum[j][idx + 1]
                    ],
                ),
                num_nodes=self.rgmol_n_node[j][idx],
            )
            for j in range(self.rgmol_max_cnt)
        ]

        for j in range(self.rgmol_max_cnt):
            rg[j].ndata["attr"] = torch.from_numpy(
                self.rgmol_node_attr[j][
                    self.rgmol_n_csum[j][idx] : self.rgmol_n_csum[j][idx + 1]
                ]
            ).float()
            rg[j].edata["edge_attr"] = torch.from_numpy(
                self.rgmol_edge_attr[j][
                    self.rgmol_e_csum[j][idx] : self.rgmol_e_csum[j][idx + 1]
                ]
            ).float()




        label = self.y[idx]

        return *g1, *g2, label

    def __len__(self):
        return self.y.shape[0]
