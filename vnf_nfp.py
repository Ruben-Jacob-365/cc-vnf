import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

# create network model
while True:
    G = nx.erdos_renyi_graph(10, 0.5, seed=123)
    if nx.is_connected(G):
        break

for node in G.nodes():
    G.nodes[node]['VM_capacity'] = random.randint(2, 5)
    G.nodes[node]['processing_rate'] = np.random.uniform(1.5, 2.5)

for edge in G.edges():
    G.edges[edge]['delay'] = np.random.uniform(1, 10)

# Chains
F = ['FW', 'IDS', 'LB', 'NAT', 'VPN']
service_rate = 0.5

num_chains = 30
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

Nf = {f: 0 for f in F}

for chain in chains:
    for f in chain['functions']:
        Nf[f] = Nf.get(f, 0) + chain['arrival_rate']/service_rate

# Print Nf for debugging
print("\n--- Nf Values ---")
for f in Nf:
    print(f"Function {f}: Nf = {Nf[f]:.4f}")


# deploy vnfs
alpha = 0.5 
Lf = { (f1, f2): 1 if f1 != f2 and np.random.rand() < 0.5 else 0 for f1 in F for f2 in F } 

deployed_vnfs = { f: [] for f in F }
node_capacity = { v: G.nodes[v]['VM_capacity'] for v in G.nodes() }

# Continue while any function has unmet demand
while any(len(deployed_vnfs[f]) < int(np.ceil(Nf[f])) for f in F):

    function_to_deploy = max((f for f in F if len(deployed_vnfs[f]) < int(np.ceil(Nf[f]))),
                             key=lambda f: Nf[f] - len(deployed_vnfs[f]))

    candidate_nodes = [v for v in G.nodes() if node_capacity[v] > 0]
    if not candidate_nodes:
        print("❌ No more nodes with available capacity.")
        break

    best_node = max(candidate_nodes, key=lambda v: scores[function_to_deploy][v])

    # Deploy instance
    deployed_vnfs[function_to_deploy].append(best_node)
    node_capacity[best_node] -= 1

    print(f"✅ Deployed function {function_to_deploy} on node {best_node}")

    # Update scores for same function (decay based on distance)
    for v in G.nodes():
        if v == best_node:
            d = 1
        else:
            try:
                d = nx.shortest_path_length(G, source=best_node, target=v)
            except nx.NetworkXNoPath:
                continue
        scores[function_to_deploy][v] *= alpha ** (1 / (d))  # avoid division by zero

    # Update scores for different functions based on parallelism
    for f_prime in F:
        if f_prime == function_to_deploy:
            continue
        for v in G.nodes():
            if v == best_node:
                d = 1
            else:
                try:
                    d = nx.shortest_path_length(G, source=best_node, target=v)
                except nx.NetworkXNoPath:
                    continue
            boost = (1 + Lf.get((function_to_deploy, f_prime), 0)) ** (1 / (d))
            scores[f_prime][v] *= boost

# Print deployed VNF instances
print("\n--- VNF Deployment ---")
for f in deployed_vnfs:
    print(f"Function {f}: Deployed at nodes {deployed_vnfs[f]}")

# instance assignment
# def assign_instances(G, chains, deployed_vnfs):
#     assignments = {}

    
#     for chain in chains:
#         print(f"\nProcessing Chain: {chain['source']} → {chain['destination']} with functions {chain['functions']}")   
#         assigned_path = []
#         total_delay = 0

#         for f in chain['functions']:
#             possible_nodes = deployed_vnfs.get(f, [])

#             if not possible_nodes:
#                 print(f"Function {f} not deployed")
#                 possible_nodes = list(G.nodes())

#             best_node = min(possible_nodes, key=lambda x: nx.shortest_path_length(G, source=assigned_path[-1] if assigned_path else chain['source'], target=x, weight='delay'))
#             assigned_path.append(best_node)

#             delay = 0

#             if len(assigned_path) > 1:
#                 prev_node = assigned_path[-2]
#                 delay = nx.shortest_path_length(G, source=prev_node, target=best_node, weight='delay')
#                 total_delay += delay
#                 print(f"→ Assigned {f} to Node {best_node} (Delay: {delay} ms)")

#         print(f"✅ Total delay for chain: {total_delay} ms")
        
#         assignments[chain['source'], chain['destination']] = {"path":assigned_path, "delay":total_delay}

#     return assignments

# assignments = assign_instances(G, chains, deployed_vnfs)

# # plot
# import matplotlib.cm as cm

# def draw_network_with_paths(G, assignments):
#     pos = nx.spring_layout(G)  # Layout for visualization
#     plt.figure(figsize=(12, 8))

#     # Draw base network
#     nx.draw(G, pos, with_labels=True, node_size=500, node_color="lightgray", edge_color="gray")

#     # Generate a unique color for each chain
#     num_chains = len(assignments)
#     colors = cm.rainbow(np.linspace(0, 1, num_chains))

#     for i, ((src, dst), data) in enumerate(assignments.items()):
#         print(f"Service Chain {src} → {dst}: Path {data['path']}")
#         path = data["path"]

#         # Ensure at least one edge exists in the path
#         if len(path) < 2:
#             print(f"⚠️ Warning: Chain {src} → {dst} has an incomplete path: {path}")
#             continue  # Skip if no valid edges

#         edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        
#         # Draw the path in a unique color
#         nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=[colors[i]], width=2.5)

#     plt.title("Service Chains and VNF Assignments")
#     plt.show()

# # Call the function
# draw_network_with_paths(G, assignments)

