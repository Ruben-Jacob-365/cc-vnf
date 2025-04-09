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

num_chains = 3
chains = []

# generate random chains
for i in range(num_chains):
    src, dst = np.random.choice(list(G.nodes), 2, replace=False)
    chain_length = np.random.randint(2,5)
    functions = np.random.choice(F, chain_length, replace=False)
    arrival_rate = np.random.uniform(0.3, 0.5)
    chains.append({"source":src, "destination":dst, "functions":functions, "arrival_rate":arrival_rate})
    # print("Chain:",end=" ")
    # for j in functions:
    #     print(j,end=" -> ")
    # print()
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

#calculate mean likelihood
n = parallel_likelihood.shape[0]
masked_array = parallel_likelihood[~np.eye(n, dtype=bool)]
mean_likelihood = masked_array.mean()

# Display as a DataFrame for better readability (optional)
#display 2d matrix
print("\n--- parallelizable likelihood ---")
import pandas as pd
parallel_df = pd.DataFrame(parallel_likelihood, index=F, columns=F)
print(parallel_df)


# calculate score
def compute_scores(G, chains, F):
    scores = {f: {v:0 for v in G.nodes()} for f in F}
    distance_scores = {f: {v: 0 for v in G.nodes()} for f in F}
    cluster_scores = {f: {v: 0 for v in G.nodes()} for f in F}
    for chain in chains:
        path = nx.shortest_path(G, chain['source'], target=chain['destination'], weight='delay')
        best_nodes = [-1 for k in range(len(chain['functions']))]
        for i,f in enumerate(chain['functions']):
            best_position = int(len(path) * (i+1 / len(chain['functions'])))
            best_nodes[i] = best_position
        # print(best_nodes)
        for i, f in enumerate(chain['functions']):
            best_position = best_nodes[i]
            best_next_position = best_nodes[i+1] if i+1<len(best_nodes) else len(path)-1

            for v in path:
                # print(path.index(v), best_position, best_next_position)
                distance_scores [f][v]= 1 / ((abs(path.index(v) - best_position) + abs(path.index(v)-best_next_position)))

    #cluster scoring
    cc_scores={v:nx.clustering(G, v) for v in G.nodes()}
    cc_mean = np.mean(list(cc_scores.values()))
    for v in G.nodes:
        for f1 in range(len(F)):
            for f2 in range(f1,len(F)):
                func=F[f1]
                cluster_scores[func][v]+=(cc_scores[v]-cc_mean)*(parallel_likelihood[f1][f2]-mean_likelihood)

    for f in scores:
        for v in scores[f]:
            scores[f][v]=distance_scores[f][v]*(1+cluster_scores[f][v])

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
Lf = { (f1, f2): parallel_likelihood[func_index[f1]][func_index[f2]] for f1 in F for f2 in F }

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
    remaining_capacity = {v: mu for v in G.nodes()}
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
            candidate_utilization = max((chain['arrival_rate'] / remaining_capacity[v]) if remaining_capacity[v] > 0 else float('inf') for v in path)
            if candidate_utilization < best_candidate_load:
                best_candidate = (path, comp_time)
                best_candidate_score = candidate_utilization

        
        if best_candidate is None:
            print("⚠️ No valid candidate found.")
            continue

        chosen_path, final_time = best_candidate
        last_node = chosen_path[-1]
        try:
            dest_delay = nx.shortest_path_length(G, source=last_node, target=chain['destination'], weight='delay')
            final_time += dest_delay
            chosen_path.append(chain['destination'])
        except nx.NetworkXNoPath:
            print(f"⚠️ No path from last node {last_node} to destination {chain['destination']}. Skipping chain.")
            continue

        for node in chosen_path[:-1]:  # Exclude destination node from workload update
            workload[node] += chain['arrival_rate']
            remaining_capacity[node] = max(0, remaining_capacity[node] - chain['arrival_rate'])

        assignments[(chain['source'], chain['destination'])] = {
            "path": chosen_path,
            "delay": final_time,
            "workload": {node: workload.get(node, 0) for node in chosen_path},
            "remaining_capacity": {node: remaining_capacity.get(node, mu) for node in chosen_path},
            "candidates": candidate_paths,
        }
        print(f"✅ Assigned chain from {chain['source']} to {chain['destination']}:")
        print(f"    Path: {chosen_path}, Total Delay: {final_time:.2f} ms, Max Utilization: {best_candidate_score:.2f}")
    
    return assignments

assignments = assign_instances(G, chains, deployed_vnfs, mu=0.5, K=K, T_func=T_func, T_proc=T_proc)

# Visualization
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def draw_detailed_chain_plot(G, chain_key, chain_data):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 8))
    plt.title(f"Service Chain {chain_key[0]} → {chain_key[1]}\nSelected Path: {chain_data['path']}, Total Delay: {chain_data['delay']:.2f} ms")

    # Base graph
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color="lightgray")
    nx.draw_networkx_edges(G, pos, edge_color="lightgray", width=1)
    nx.draw_networkx_edge_labels(G, pos,
        edge_labels={(u, v): f"{G.edges[(u, v)]['delay']:.1f}" for u, v in G.edges()},
        font_size=7
    )

    # Node labels with workload/remaining capacity
    node_labels = {v: f"{v}\n(load: {chain_data['workload'].get(v, 0):.2f}, rem: {chain_data['remaining_capacity'].get(v, 0):.2f})"
                   for v in chain_data['path']}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9)

    # Draw top K candidate paths in light colors
    for path, _ in chain_data.get("candidates", []):
        candidate_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=candidate_edges, edge_color="skyblue", style="dashed", width=1.5)

    # Draw final chosen path in red
    chosen_path = chain_data["path"]
    final_edges = [(chosen_path[i], chosen_path[i+1]) for i in range(len(chosen_path)-1)]
    nx.draw_networkx_nodes(G, pos, nodelist=chosen_path, node_color="red", node_size=700)
    nx.draw_networkx_edges(G, pos, edgelist=final_edges, edge_color="red", width=3)

    # Stage annotation
    stage_labels = {node: f"Stage {i}" for i, node in enumerate(chosen_path)}
    nx.draw_networkx_labels(G, pos, labels=stage_labels, font_color="white", font_size=9)

    plt.axis("off")
    plt.tight_layout()
    plt.show()

def draw_network_with_separate_plots(G, assignments):
    for i, ((src, dst), data) in enumerate(assignments.items()):
        print(f"Service Chain {src} → {dst}: Path {data['path']} with Total Delay {data['delay']:.2f} ms")
        draw_detailed_chain_plot(G, (src, dst), data)

# Call the visualization function.
draw_network_with_separate_plots(G, assignments)