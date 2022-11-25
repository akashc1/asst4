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

    int numNodes = num_nodes(g);
    double equal_prob = 1.0 / numNodes;

    bool converged = false;
    double* scores1 = (double*) malloc(sizeof(double) * numNodes);
    double* scores2 = (double*) malloc(sizeof(double) * numNodes);

    Vertex* no_outgoing_vtx = (Vertex*) malloc(sizeof(Vertex) * numNodes);
    int no_outgoing_count{0};
    #pragma omp parallel for
    for (int i = 0; i < numNodes; i++) {
        scores1[i] = equal_prob;

        if (!outgoing_size(g, i)) {
            int idx;
            #pragma omp critical
            {
                idx = no_outgoing_count++;
            }
            no_outgoing_vtx[idx] = i;
        }
    }

    double global_diff;
    while (!converged) {

        // accumulate scores for vertices with no outgoing edges
        double no_outgoing_total_score{0};
        #pragma omp parallel for reduction(+:no_outgoing_total_score)
        for (int i = 0; i < no_outgoing_count; i++) {
            no_outgoing_total_score += scores1[no_outgoing_vtx[i]];
        }

        no_outgoing_total_score = damping * no_outgoing_total_score / numNodes;

        #pragma omp parallel for
        for (int i = 0; i < numNodes; i++) {
            double new_score{0};

            if (incoming_size(g, i)) {
                // sum over all nodes vj reachable from incoming edges
                const Vertex* start = incoming_begin(g, i);
                const Vertex* end = incoming_end(g, i);
                for (const Vertex* in_v = start; in_v != end; in_v++) {
                    new_score += scores1[*in_v] / outgoing_size(g, *in_v);
                }
            }

            scores2[i] = no_outgoing_total_score + (damping * new_score) + (1 - damping) / numNodes;
        }

        // accumulate global diff, update old scores
        global_diff = 0;
        #pragma omp parallel for reduction(+:global_diff)
        for (int i = 0; i < numNodes; i++) {
            global_diff += abs(scores2[i] - scores1[i]);
            scores1[i] = scores2[i];
        }

        converged = global_diff < convergence;
    }

    // write back solution
    #pragma omp parallel for
    for (int i = 0; i < numNodes; i++) {
        solution[i] = scores1[i];
    }

    free(scores1);
    free(scores2);
    free(no_outgoing_vtx);
}
