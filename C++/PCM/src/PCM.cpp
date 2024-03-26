#include <PCM/PCM.hpp>

#include <fmc/findClique.h>
#include <fmc/graphIO.h>

void build_CGraphIO_from_Matrix(const Eigen::MatrixXi& data, CGraphIO& graph) {
  int n = data.rows();

  int edge_index = 0;

  for (int i = 0; i < n; i++) {
    graph.m_vi_Vertices.push_back(edge_index);

    const Eigen::VectorXi& edge_list_i = data.col(i);
    for (int j = 0; j < n; j++) {
      if (i != j) {
        if (edge_list_i(j)) {
          graph.m_vi_Edges.push_back(j);
          edge_index++;
        }
      }
    }
  }
  graph.m_vi_Vertices.push_back(edge_index);

  graph.CalculateVertexDegrees();
}

std::vector<bool> PCM::PattabiramanMaxCliqueSolverExact::find_max_clique(
    const Eigen::MatrixXi& adjacency_matrix) const {
  CGraphIO graph;
  build_CGraphIO_from_Matrix(adjacency_matrix, graph);

  std::vector<int> max_clique_data;

  maxClique(graph, 0, max_clique_data);

  std::vector<bool> max_clique_map(adjacency_matrix.rows(), false);
  for (int i = 0; i < max_clique_data.size(); i++) {
    max_clique_map[max_clique_data[i]] = true;
  }

  return max_clique_map;
}

std::vector<bool> PCM::PattabiramanMaxCliqueSolverHeuristic::find_max_clique(
    const Eigen::MatrixXi& adjacency_matrix) const {
  CGraphIO graph;
  build_CGraphIO_from_Matrix(adjacency_matrix, graph);

  std::vector<int> max_clique_data;

  maxCliqueHeu(graph, max_clique_data);

  std::vector<bool> max_clique_map(adjacency_matrix.rows(), false);
  for (int i = 0; i < max_clique_data.size(); i++) {
    max_clique_map[max_clique_data[i]] = true;
  }

  return max_clique_map;
}
