#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>
#include <vector>

#include "../common/CycleTimer.h"
#include "../common/graph.h"


// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double* solution, double damping, double convergence)
{


  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

    int numNodes = num_nodes(g);
    double equal_prob = 1.0 / numNodes;

  /*
     CS149 students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
       converged = (global_diff < convergence)
     }

   */
    bool converged = false;
    double* scores1 = (double*) malloc(sizeof(double) * numNodes);
    double* scores2 = (double*) malloc(sizeof(double) * numNodes);
    double* score_diffs = (double*) malloc(sizeof(double) * numNodes);

    std::vector<Vertex> no_outgoing;
    #pragma omp parallel for schedule(dynamic, 3u)
    for (int i = 0; i < numNodes; i++) {
        if (!outgoing_size(g, i)) {
            #pragma omp critical
            {
                no_outgoing.push_back(i);
            }
        }
    }

    #pragma omp parallel for schedule(dynamic, 32)
    for (int i = 0; i < numNodes; i++) {
        scores1[i] = equal_prob;
        score_diffs[i] = 0;
    }

    double global_diff;
    while (!converged) {
        global_diff = 0;

        #pragma omp parallel for schedule(dynamic, 32)
        for (int i = 0; i < numNodes; i++) {
            double new_score = 0;

            if (incoming_size(g, i)) {
                // sum over all nodes vj reachable from incoming edges
                const Vertex* start = incoming_begin(g, i);
                const Vertex* end = incoming_end(g, i);
                for (const Vertex* in_v = start; in_v != end; in_v++) {
                    new_score += scores1[*in_v] / outgoing_size(g, *in_v);
                }
            }

            new_score = (damping * new_score) + (1 - damping) / numNodes;

            // sum over all nodes v in graph with no outgoing edges
            for (int j = 0; j < no_outgoing.size(); j++) {
                new_score += damping * scores1[no_outgoing[j]] / numNodes;
            }

            scores2[i] = new_score;
            score_diffs[i] = abs(new_score - scores1[i]);
        }

        // accumulate diffs
        #pragma omp parallel for reduction(+:global_diff) schedule(dynamic, 32)
        for (int i = 0; i < numNodes; i++) {
            global_diff += score_diffs[i];
            scores1[i] = scores2[i];
        }

        converged = global_diff < convergence;
    }

    #pragma omp parallel for schedule(dynamic, 32)
    for (int i = 0; i < numNodes; i++) {
        solution[i] = scores1[i];
    }

    free(scores1);
    free(scores2);
    free(score_diffs);
}
