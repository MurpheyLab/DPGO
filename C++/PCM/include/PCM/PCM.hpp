#ifndef __PCM_HPP
#define __PCM_HPP

#include <assert.h>

#include <Eigen/Core>
#include <vector>

namespace PCM {

///////////////////////////////////////
// Base Class of Max Clique Optimizer
///////////////////////////////////////

class MaxCliqueSolver {
 public:
  MaxCliqueSolver() {}
  virtual ~MaxCliqueSolver() {}

  // This function takes an adjacency matrix and calls the associated
  // algorithm for finding the maximum clique
  //
  // The adjacency matrix is a matrix of integers, 0 denotes no edge,
  // any other value denotes an edge.
  virtual std::vector<bool> find_max_clique(
      const Eigen::MatrixXi& adjacency_matrix) const = 0;
};

///////////////////////////////////////////////////
// Pattabiraman Max Clique Implementations
///////////////////////////////////////////////////
// These classes call the algorithms proposed in:
// B. Pattabiraman, M. M. A. Patwary, A. H. Gebremedhin, W. keng Liao, and A.
// Choudhary, “Fast algorithms for the maximum clique problem on massive graphs
// with applications to overlapping community detection,” Internet Mathematics,
// vol. 11, no. 4-5, pp. 421–448, 2015.
//
// The current implemenation (as released by the authors) only uses a single
// thread though these algorithms can be easily parallelized as explained in the
// paper above.

class PattabiramanMaxCliqueSolverExact : public MaxCliqueSolver {
 public:
  PattabiramanMaxCliqueSolverExact() : MaxCliqueSolver() {}

  // This function takes an adjacency matrix and calls the associated
  // algorithm for finding the maximum clique
  //
  // The adjacency matrix is a matrix of integers, 0 denotes no edge,
  // any other value denotes an edge.
  virtual std::vector<bool> find_max_clique(
      const Eigen::MatrixXi& adjacency_matrix) const override;
};

class PattabiramanMaxCliqueSolverHeuristic : public MaxCliqueSolver {
 public:
  PattabiramanMaxCliqueSolverHeuristic() : MaxCliqueSolver() {}

  // This function takes an adjacency matrix and calls the associated
  // algorithm for finding the maximum clique
  //
  // The adjacency matrix is a matrix of integers, 0 denotes no edge,
  // any other value denotes an edge.
  virtual std::vector<bool> find_max_clique(
      const Eigen::MatrixXi& adjacency_matrix) const override;
};

/////////////////////////////////////////
// Consistency Evaluator Base Class
/////////////////////////////////////////
class ConsistencyEvaluator {
 public:
  ConsistencyEvaluator() {}
  virtual ~ConsistencyEvaluator() {}

 public:
  // This function should return the consistency of
  // the i-th and j-th measurement
  virtual bool evaluate_consistency(int i, int j) const = 0;

  // This function should return the consistency of
  // the i-th measurement with all measurements from 0 upto the (i-1)th
  // measurement in that order
  //
  // Can be parallelized if desired
  virtual std::vector<bool> evaluate_consistency_with_prior_measurements(
      int i) {
    std::vector<bool> consistency(i);
    for (int j = 0; j < i; j++) {
      consistency[j] = this->evaluate_consistency(i, j);
    }
    return consistency;
  }

  // This function should return the consistency of the measurements in
  // the range i_min to i_max inclusive as
  // vectors of booleans of the form returned by:
  // std::vector<bool> evaluate_consistency(i)
  //
  // Can be parallelized if desired
  // Currently not used in PCM Solver Implementation
  virtual std::vector<std::vector<bool> > evaluate_consistency_for_range(
      int i_min, int i_max) {
    std::vector<std::vector<bool> > consistency_lists(i_max - i_min + 1);
    for (int i = i_min; i <= i_max; i++) {
      consistency_lists[i] = evaluate_consistency_with_prior_measurements(i);
    }
    return consistency_lists;
  }
};

/////////////////////////////////////////
// PCM Solver
/////////////////////////////////////////

// The template parameters specify the algorithm that will be used
// to find the maximum clique of the graph
// and the function that will be used to evaluate consistency.
//
// The MaxCliqueSolverT template class must be derived from MaxCliqueSolver
//
// The ConsistencyEvaluatorT object must be callable with two integers
// specifying the indices of two measurements and return a bool.
// It should also be callable with a single integer (i) and return a vector
// of bools with length i-1 specifying the consistency of the measurement
// i with all measurements before it.
template <class MaxCliqueSolverT, class ConsistencyEvaluatorT>
class PCMSolver {
  // Data Members
 private:
  MaxCliqueSolverT& mcs;  // Maximum clique solver

  ConsistencyEvaluatorT& consistency_evaluator;

  int num_measurements;

  std::vector<std::vector<bool> > consistency_data;

  // Function Members
 private:
  void evaluate_consistency_for_new_measurement() {
    int new_measurement = num_measurements;
    std::vector<bool> consistency =
        consistency_evaluator.evaluate_consistency_with_prior_measurements(
            new_measurement);

    consistency_data.push_back(consistency);
  }

  void build_adjacency_matrix(Eigen::MatrixXi& matrix) {
    matrix.resize(this->num_measurements, this->num_measurements);
    for (int i = 0; i < this->num_measurements; i++) {
      matrix(i, i) = 1;
      for (int j = 0; j < i; j++) {
        int weight = (this->consistency_data[i][j]) ? 1 : 0;
        matrix(i, j) = weight;
        matrix(j, i) = weight;
      }
    }
  }

 public:
  PCMSolver(MaxCliqueSolverT& mcs_,
            ConsistencyEvaluatorT& consistency_evaluator_)
      : mcs(mcs_),
        consistency_evaluator(consistency_evaluator_),
        num_measurements(0) {
    static_assert(std::is_base_of<MaxCliqueSolver, MaxCliqueSolverT>::value,
                  "MaxCliqueSolverT type not derived from MaxCliqueSolver");
    static_assert(
        std::is_base_of<ConsistencyEvaluator, ConsistencyEvaluatorT>::value,
        "ConsistencyEvaluatorT type not derived from ConsistencyEvaluator");
  }

  std::vector<bool> solve_pcm() {
    Eigen::MatrixXi adjacency_matrix;
    this->build_adjacency_matrix(adjacency_matrix);
    return this->mcs.find_max_clique(adjacency_matrix);
  }

  void add_measurements(int num_measurements_to_add) {
    for (int i = 0; i < num_measurements_to_add; i++) {
      this->evaluate_consistency_for_new_measurement();
      this->num_measurements++;
    }
  }
  int get_total_measurements() { return num_measurements; }
};

}  // namespace PCM
#endif  //__PCM_HPP
