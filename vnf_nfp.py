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

    candidate_nodes = [v for v in G.nodes() if node_capacity[v] > 0 and v not in deployed_vnfs[function_to_deploy]]
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
T_func = 5
T_proc = 2
K = 3

def assign_instances(G, chains, deployed_vnfs, mu, K=3, T_func=5, T_proc=2):
    # sort chains in descending order of arrival rate
    chains_sorted = sorted(chains, key=lambda x: x['arrival_rate'], reverse=True)

    workload = {v: 0 for v in G.nodes()}
    assignments = {}

    for chain in chains_sorted:
        print(f"\nProcessing chain from {chain['source']} to {chain['destination']} with functions {chain['functions']} and arrival rate {chain['arrival_rate']:.2f}")

        m = len(chain['functions'])
        # initialization for stage 0
        candidate_paths = [ ([chain['source']], 0)]

        # process each stage j = 1 to m
        for j in range(1, m+1):
            current_function = chain['functions'][j-1]
            new_candidates = []

            for path, compt_time in candidate_paths:
                last_node = path[-1]

                available_instances = deployed_vnfs.get(current_function, [])
                if not available_instances:
                    print(f"No available instances for function {current_function}.")
                    continue

                for instance in available_instances:
                    try:
                        tx_delay = nx.shortest_path_length(G, source=last_node, target=instance, weight='delay')
                    except nx.NetworkXNoPath:
                        continue

                    # compute stage delay
                    stage_delay = T_func + T_proc + tx_delay
                    new_comp_time = compt_time + stage_delay
                    new_path = path + [instance]
                    new_candidates.append((new_path, new_comp_time))
            
            if not new_candidates:
                print(f"⚠️ Warning: Could not extend candidate paths at stage {j} for function {current_function}.")
                candidate_paths = []
                break

            new_candidates.sort(key=lambda x: x[1])
            # keep K best candidates
            candidate_paths = new_candidates[:K]
            print(f"Stage {j} ({current_function}): Keeping {len(candidate_paths)} candidates.")
        
        if not candidate_paths:
            print("⚠️ No valid paths found for this chain.")
            continue

        best_candidate = None
        best_candidate_load = float('inf')
        for path, comp_time in candidate_paths:
            max_load = max(workload[v] for v in path)
            if max_load < best_candidate_load:
                best_candidate = (path, comp_time)
                best_candidate_load = max_load
        
        if best_candidate is None:
            print("⚠️ No valid candidate found.")
            continue

        chosen_path, final_time = best_candidate

        for node in chosen_path:
            workload[node] += chain['arrival_rate']

        assignments[(chain['source'], chain['destination'])] = {
            "path": chosen_path,
            "delay": final_time,
            "workload": {node: workload[node] for node in chosen_path}
        }
        print(f"✅ Assigned chain from {chain['source']} to {chain['destination']}:")
        print(f"    Path: {chosen_path}, Total Delay: {final_time:.2f} ms, Max Workload: {best_candidate_load:.2f}")
    
    return assignments

assignments = assign_instances(G, chains, deployed_vnfs, mu=0.5, K=K, T_func=T_func, T_proc=T_proc)

# Visualization
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def draw_detailed_chain_plot(G, chain_key, chain_data):
    """
    Draws a detailed plot for a specific service chain.
    
    chain_key: Tuple (source, destination)
    chain_data: Dictionary with keys "path", "delay", "workload"
    """
    pos = nx.spring_layout(G, seed=42)
    
    plt.figure(figsize=(10, 8))
    plt.title(f"Service Chain {chain_key[0]} → {chain_key[1]}\nPath: {chain_data['path']}, Total Delay: {chain_data['delay']:.2f} ms")
    
    # Draw the entire network in light gray
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color="lightgray")
    nx.draw_networkx_edges(G, pos, edge_color="lightgray", width=1)
    
    # Draw nodes with workload info using the workload from chain_data
    # Note: This workload only covers nodes on the chosen path for this chain.
    node_labels = {v: f"{v}\n(load: {chain_data['workload'].get(v, 0):.2f})" for v in chain_data["path"]}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
    
    # Draw edge labels with the transmission delay on each edge.
    edge_labels = {(u, v): f"{G.edges[(u, v)]['delay']:.1f}" for u, v in G.edges() if 'delay' in G.edges[(u, v)]}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    # Highlight the candidate path for the chain with a bold color (red)
    chain_path = chain_data["path"]
    chain_edges = [(chain_path[i], chain_path[i+1]) for i in range(len(chain_path) - 1)]
    nx.draw_networkx_nodes(G, pos, nodelist=chain_path, node_color="red", node_size=700)
    nx.draw_networkx_edges(G, pos, edgelist=chain_edges, edge_color="red", width=3)
    
    # Optionally annotate nodes in the chain with their stage index.
    stage_labels = {node: f"Stage {i}" for i, node in enumerate(chain_path)}
    nx.draw_networkx_labels(G, pos, labels=stage_labels, font_color="white", font_size=10)
    
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def draw_network_with_separate_plots(G, assignments):
    # For each chain, use its own workload info stored in the assignments
    for i, ((src, dst), data) in enumerate(assignments.items()):
        print(f"Service Chain {src} → {dst}: Path {data['path']} with Total Delay {data['delay']:.2f} ms")
        # Pass only the per-chain workload stored in data
        draw_detailed_chain_plot(G, (src, dst), data)

# Call the function to create separate plots for each service chain.
draw_network_with_separate_plots(G, assignments)
