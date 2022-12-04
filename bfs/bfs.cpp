#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set* list) {
    list->count = 0;
}

void vertex_set_init(vertex_set* list, int count) {
    list->max_vertices = count;
    list->vertices = (int*)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

void vertex_inclusion_set_clear(vertex_inclusion_set* list) {
    list->count = 0;
    #pragma omp parallel for
    for (int i = 0; i < list->max_vertices; i++) {
        list->vertices[i] = false;
    }
}

void vertex_inclusion_set_init(vertex_inclusion_set* list, int count) {
    list->max_vertices = count;
    list->vertices = (bool*)malloc(sizeof(bool) * list->max_vertices);
    vertex_inclusion_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set* frontier,
    vertex_set* new_frontier,
    int* distances)
{
    #pragma omp parallel
    {
        vertex_set thread_frontier;
        vertex_set_init(&thread_frontier, g->num_nodes);

        #pragma omp for schedule(dynamic, 128)
        for (int i=0; i<frontier->count; i++) {

            int node = frontier->vertices[i];

            int start_edge = g->outgoing_starts[node];
            int end_edge = (node == g->num_nodes - 1)
                               ? g->num_edges
                               : g->outgoing_starts[node + 1];

            // attempt to add all neighbors to the new frontier
            for (int neighbor=start_edge; neighbor<end_edge; neighbor++) {
                int outgoing = g->outgoing_edges[neighbor];

                if (distances[outgoing] == NOT_VISITED_MARKER
                        && __sync_bool_compare_and_swap(distances + outgoing, NOT_VISITED_MARKER, distances[node] + 1)) {

                    thread_frontier.vertices[thread_frontier.count++] = outgoing;
                }
            }
        }

        int write_start = new_frontier->count;
        while (!(new_frontier->count == write_start
                    && __sync_bool_compare_and_swap(&new_frontier->count, write_start, write_start + thread_frontier.count))) {
            write_start = new_frontier->count;
        }
        for (int j = write_start; j < write_start + thread_frontier.count; j++) {
            new_frontier->vertices[j] = thread_frontier.vertices[j - write_start];
        }

        free(thread_frontier.vertices);
    }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution* sol) {

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i=0; i<graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

void bottom_up_step(
    Graph g,
    vertex_inclusion_set* frontier,
    vertex_inclusion_set* new_frontier,
    int* distances)
{
    #pragma omp parallel for schedule(dynamic, 256)
    for (int n = 0; n < g->num_nodes; n++) {
        // no need to consider already visited nodes
        if (distances[n] != NOT_VISITED_MARKER) {
            continue;
        }

        int start_edge = g->incoming_starts[n];
        int end_edge = (n== g->num_nodes - 1) ? g->num_edges : g->incoming_starts[n + 1];

        // check for nodes which lead to this one, use their distance to compute this node's
        for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
            int incoming = g->incoming_edges[neighbor];
            if (frontier->vertices[incoming]) {
                new_frontier->vertices[n] = true;
                distances[n] = distances[incoming] + 1;
                break;
            }
        }
    }
}

void bfs_bottom_up(Graph graph, solution* sol)
{
    vertex_inclusion_set list1, list2;
    vertex_inclusion_set_init(&list1, graph->num_nodes);
    vertex_inclusion_set_init(&list2, graph->num_nodes);

    vertex_inclusion_set* frontier = &list1;
    vertex_inclusion_set* new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i=0; i<graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[ROOT_NODE_ID] = true;
    frontier->count++;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0) {

        vertex_inclusion_set_clear(new_frontier);

        bottom_up_step(graph, frontier, new_frontier, sol->distances);

        vertex_inclusion_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;

        // didn't synchonously count frontier so quickly add it up here
        frontier->count = 0;
        int frontier_count = 0;
        #pragma omp parallel for reduction(+:frontier_count)
        for (int i = 0; i < frontier->max_vertices; i++) {
            if (frontier->vertices[i]) {
                frontier_count++;
            }
        }
        frontier->count = frontier_count;
    }
}

void bfs_hybrid(Graph graph, solution* sol)
{
    bfs_top_down(graph, sol);
}
