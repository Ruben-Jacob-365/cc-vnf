import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

# create network model
G = nx.erdos_renyi_graph(30,0.2, directed=False)

for node in G.nodes():
    G.nodes[node]['VM_capacity'] = random.randint(2, 5)
    G.nodes[node]['processing_rate'] = np.random.uniform(1.5, 2.5)

for edge in G.edges():
    G.edges[edge]['delay'] = np.random.uniform(1, 10)

# Chains
F = ['FW', 'IDS', 'LB', 'NAT', 'VPN']

num_chains = 10
chains = []

# generate random chains
for i in range(num_chains):
    src, dst = np.random.choice(list(G.nodes), 2, replace=False)
    chain_length = np.random.randint(2,5)
    functions = np.random.choice(F, chain_length, replace=False)
    arrival_rate = np.random.uniform(0.3, 0.5)
    chains.append({"source":src, "destination":dst, "functions":functions, "arrival_rate":arrival_rate})
    print("Chain:",end=" ")
    for j in functions:
        print(j,end=" -> ")
    print()
# 2d matrix with P(fi,fj) values
parallel_matrix = np.zeros((len(F), len(F)), dtype=int)

for i in range(len(F)):
    for j in range(i, len(F)):
        if i == j:
            parallel_matrix[i][j] = 0  # A function is always parallelizable with itself
        else:
            val = random.randint(0, 1)
            parallel_matrix[i][j] = val
            parallel_matrix[j][i] = val




#for parallel likelihood

parallel_likelihood = np.zeros((len(F), len(F)))

# Create a mapping from function name to index for easy lookup
func_index = {fname: idx for idx, fname in enumerate(F)}

# Iterate over each chain
for chain in chains:
    funcs = chain["functions"]
    # Check all pairs of functions in the chain
    for i in range(len(funcs)):
        for j in range(i+1, len(funcs)):
            fi, fj = funcs[i], funcs[j]
            idx_i, idx_j = func_index[fi], func_index[fj]
            # Check if the pair is parallelizable
            if parallel_matrix[idx_i][idx_j] == 1:
                parallel_likelihood[idx_i][idx_j] += 1/len(chains)
                parallel_likelihood[idx_j][idx_i] += 1/len(chains)  # Keep it symmetric

# Display as a DataFrame for better readability (optional)
#display 2d matrix
print("\n--- parallelizable likelihood ---")
import pandas as pd
parallel_df = pd.DataFrame(parallel_likelihood, index=F, columns=F)
print(parallel_df)


# calculate score
def compute_scores(G, chains, F):
    scores = {f: {v:0 for v in G.nodes()} for f in F}

    for chain in chains:
        path = nx.shortest_path(G, chain['source'], target=chain['destination'], weight='delay')
        for i,f in enumerate(chain['functions']):
            best_position = int(len(path) * (i / len(chain['functions'])))
            best_node = path[best_position]

            for v in path:
                distance_factor = 1 / (abs(path.index(v) - best_position) + 1)
                scores[f][v] += distance_factor * (1 + nx.clustering(G, v))

    # Print scores for debugging
    print("\n--- Score Calculations ---")
    for f in F:
        sorted_scores = sorted(scores[f].items(), key=lambda x: x[1], reverse=True)
        print(f"\nFunction: {f}")
        for node, score in sorted_scores[:5]:  # Show top 5 nodes
            print(f"Node {node}: Score = {score:.4f}")

    return scores

scores = compute_scores(G, chains, F)

# deploy vnfs
deployed_vnfs = { f: [] for f in F}

for f in F:
    available_nodes = set(G.nodes())

    for _ in range(np.random.randint(1,4)):
        if not available_nodes:
            break

        best_node = max(available_nodes, key=lambda x: scores[f][x])
        deployed_vnfs[f].append(best_node)
        available_nodes.remove(best_node)

        # reduce score for same function nearby
        for neighbor in nx.neighbors(G, best_node):
            scores[f][neighbor] *= 0.7 # decay factor

        # Increase score for parallelizeable functions
        for f_prime in F:
            if np.random.rand() < 0.5: #assunming 50% chance of parallelization
                for neighbor in nx.neighbors(G, best_node):
                    scores[f_prime][neighbor] *= 1.2 # boost factor

# Print deployed VNF instances
print("\n--- VNF Deployment ---")
for f in deployed_vnfs:
    print(f"Function {f}: Deployed at nodes {deployed_vnfs[f]}")

# instance assignment
def assign_instances(G, chains, deployed_vnfs):
    assignments = {}


    for chain in chains:
        print(f"\nProcessing Chain: {chain['source']} → {chain['destination']} with functions {chain['functions']}")
        assigned_path = []
        total_delay = 0

        for f in chain['functions']:
            possible_nodes = deployed_vnfs.get(f, [])

            if not possible_nodes:
                print(f"Function {f} not deployed")
                possible_nodes = list(G.nodes())

            best_node = min(possible_nodes, key=lambda x: nx.shortest_path_length(G, source=assigned_path[-1] if assigned_path else chain['source'], target=x, weight='delay'))
            assigned_path.append(best_node)

            delay = 0

            if len(assigned_path) > 1:
                prev_node = assigned_path[-2]
                delay = nx.shortest_path_length(G, source=prev_node, target=best_node, weight='delay')
                total_delay += delay
                print(f"→ Assigned {f} to Node {best_node} (Delay: {delay} ms)")

        print(f"✅ Total delay for chain: {total_delay} ms")

        assignments[chain['source'], chain['destination']] = {"path":assigned_path, "delay":total_delay}

    return assignments

assignments = assign_instances(G, chains, deployed_vnfs)

# plot
import matplotlib.cm as cm

def draw_network_with_paths(G, assignments):
    pos = nx.spring_layout(G)  # Layout for visualization
    plt.figure(figsize=(12, 8))

    # Draw base network
    nx.draw(G, pos, with_labels=True, node_size=500, node_color="lightgray", edge_color="gray")

    # Generate a unique color for each chain
    num_chains = len(assignments)
    colors = cm.rainbow(np.linspace(0, 1, num_chains))

    for i, ((src, dst), data) in enumerate(assignments.items()):
        print(f"Service Chain {src} → {dst}: Path {data['path']}")
        path = data["path"]

        # Ensure at least one edge exists in the path
        if len(path) < 2:
            print(f"⚠️ Warning: Chain {src} → {dst} has an incomplete path: {path}")
            continue  # Skip if no valid edges

        edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]

        # Draw the path in a unique color
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=[colors[i]], width=2.5)

    plt.title("Service Chains and VNF Assignments")
    plt.show()

# Call the function
draw_network_with_paths(G, assignments)

