import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import re,ast

all_chains = []
pattern = re.compile(
    r"Source:\s*(\d+),\s*Destination:\s*(\d+),\s*Functions:\s*(\[.*?\]),\s*Arrival\s*Rate:\s*([\d.]+)"
)
with open("chains.txt") as f:
    for line in f:
        m = pattern.match(line.strip())
        if not m:
            continue
        src   = int(m.group(1))
        dst   = int(m.group(2))
        funcs = ast.literal_eval(m.group(3))
        arr   = float(m.group(4))
        all_chains.append({
            "source": src,
            "destination": dst,
            "functions": funcs,
            "arrival_rate": arr
        })

print(all_chains)

while True:
    G = nx.erdos_renyi_graph(10, 0.5)
    if nx.is_connected(G):
        break
for v in G.nodes():
    G.nodes[v]['VM_capacity'] = random.randint(2, 5)
    G.nodes[v]['processing_rate'] = np.random.uniform(1.5, 2.5)
for u, v in G.edges():
    G.edges[u, v]['delay'] = np.random.uniform(1, 10)

# Define constants and function list 
F = ['FW', 'IDS', 'LB', 'NAT', 'VPN']
service_rate = 0.5 
alpha = 0.5       
T_func = 5     
T_proc = 2        
K = 3               

# Compute parallel likelihood matrix 
func_index = {f:i for i,f in enumerate(F)}
parallel_matrix = np.zeros((len(F),len(F)), dtype=int)
for i in range(len(F)):
    for j in range(i+1,len(F)):
        val = random.randint(0,1)
        parallel_matrix[i,j] = val
        parallel_matrix[j,i] = val
# compute likelihood
parallel_likelihood = np.zeros((len(F),len(F)))
for chain in all_chains:
    funcs = chain['functions']
    for i in range(len(funcs)):
        for j in range(i+1, len(funcs)):
            fi, fj = funcs[i], funcs[j]
            pi, pj = func_index[fi], func_index[fj]
            if parallel_matrix[pi,pj] == 1:
                parallel_likelihood[pi,pj] += 1/len(all_chains)
                parallel_likelihood[pj,pi] += 1/len(all_chains)
# mean off-diagonal
n = len(F)
mask = ~np.eye(n, dtype=bool)
mean_likelihood = parallel_likelihood[mask].mean()

# Score computation (distance + clustering) 
def compute_scores(G, chains):
    # initialize
    distance_scores = {f:{v:0 for v in G.nodes()} for f in F}
    cluster_scores  = {f:{v:0 for v in G.nodes()} for f in F}
    # distance-based
    for chain in chains:
        path = nx.shortest_path(G, chain['source'], chain['destination'], weight='delay')
        L = len(chain['functions'])
        best_nodes = []
        for i,f in enumerate(chain['functions']):
            pos = int(len(path) * ((i+1)/L))
            best_nodes.append(pos)
        for i,f in enumerate(chain['functions']):
            a = best_nodes[i]
            b = best_nodes[i+1] if i+1 < len(best_nodes) else (len(path)-1)
            for v in path:
                idx = path.index(v)
                distance_scores[f][v] += 1 / ((abs(idx - a) + abs(idx - b)) + 1e-6)
    # clustering-based
    cc = nx.clustering(G)
    cc_mean = np.mean(list(cc.values()))
    for v in G.nodes():
        for i in range(len(F)):
            for j in range(i+1, len(F)):
                f1, f2 = F[i], F[j]
                delta = (cc[v] - cc_mean) * (parallel_likelihood[i,j] - mean_likelihood)
                cluster_scores[f1][v] += delta
                cluster_scores[f2][v] += delta
    # combine into final scores
    scores = {f:{v: distance_scores[f][v] * (1 + cluster_scores[f][v]) for v in G.nodes()} for f in F}
    # debug print top 5
    print("\n--- Score Calculations ---")
    for f in F:
        top5 = sorted(scores[f].items(), key=lambda x:x[1], reverse=True)[:5]
        print(f"{f}: " + ", ".join(f"Node {v}={s:.3f}" for v,s in top5))
    return scores

# VNF deployment extension 
def extend_deployment(G, scores, deployed, node_cap, Nf):
    for f in F:
        needed = int(np.ceil(Nf[f])) - len(deployed[f])
        for _ in range(max(0,needed)):
            cands = [v for v in G.nodes() if node_cap[v]>0 and v not in deployed[f]]
            if not cands:
                print(f"❌ Cannot deploy more {f}")
                break
            vstar = max(cands, key=lambda v: scores[f][v])
            deployed[f].append(vstar)
            node_cap[vstar] -= 1
            # update same-function
            for v in G.nodes():
                d = 1 if v==vstar else nx.shortest_path_length(G, vstar, v)
                scores[f][v] *= alpha ** (1/d)
            # update others
            for f2 in F:
                if f2==f: continue
                i1, i2 = func_index[f], func_index[f2]
                Lf = parallel_likelihood[i1,i2]
                for v in G.nodes():
                    d = 1 if v==vstar else nx.shortest_path_length(G, vstar, v)
                    scores[f2][v] *= (1 + Lf) ** (1/d)

# Instance assignment per batch 
def assign_instances_batch(G, batch, deployed, mu, K, Tfunc, Tproc, workload, rem_capacity):
    assignments = {}
    for chain in sorted(batch, key=lambda c:c['arrival_rate'], reverse=True):
        m = len(chain['functions'])
        candidate_paths = [([chain['source']], 0)]
        # stage by stage
        for f in chain['functions']:
            new_cands = []
            for path, t0 in candidate_paths:
                last = path[-1]
                for inst in deployed[f]:
                    try:
                        tx = nx.shortest_path_length(G, last, inst, weight='delay')
                    except nx.NetworkXNoPath:
                        continue
                    t1 = t0 + Tfunc + tx + Tproc
                    new_cands.append((path+[inst], t1))
            new_cands.sort(key=lambda x:x[1])
            candidate_paths = new_cands[:K]
            print(f"Stage {f}: kept {len(candidate_paths)} candidates")
        # select by min max-util
        best = min(candidate_paths, key=lambda pt: max(
            (chain['arrival_rate']/rem_capacity[v]) if rem_capacity[v]>0 else float('inf')
            for v in pt[0]))
        path, t_final = best
        # append destination hop
        last = path[-1]
        try:
            d = nx.shortest_path_length(G, last, chain['destination'], weight='delay')
            t_final += d
            path.append(chain['destination'])
        except nx.NetworkXNoPath:
            print(f"⚠️ Cannot reach destination {chain['destination']} from {last}")
        # update workload/capacity
        for v in path[:-1]:
            workload[v] += chain['arrival_rate']
            rem_capacity[v] = max(0, rem_capacity[v] - chain['arrival_rate'])
        assignments[(chain['source'],chain['destination'])] = {
            'path': path,
            'delay': t_final
        }
        print(f"Assigned {chain['source']}→{chain['destination']}: {path}, delay={t_final:.2f}")
    return assignments

# Dynamic batching loop 
deployed_vnfs = {f:[] for f in F}
node_capacity = {v:G.nodes[v]['VM_capacity'] for v in G.nodes()}
workload = {v:0 for v in G.nodes()}
remaining_capacity = {v:service_rate for v in G.nodes()}
active_chains = []
all_assigns = {}
i = 0
while i < len(all_chains):
    batch_size = random.randint(1, 10)
    batch = all_chains[i:i+batch_size]
    i += batch_size
    print(f"\n=== New batch of size {len(batch)} ===")
    active_chains.extend(batch)

    # recompute scores on all active
    scores = compute_scores(G, active_chains)
    # recompute Nf
    Nf = {f:0 for f in F}
    for c in active_chains:
        for f in c['functions']:
            Nf[f] += c['arrival_rate']/service_rate
    # extend deployment
    extend_deployment(G, scores, deployed_vnfs, node_capacity, Nf)
    # assign this batch
    assigns = assign_instances_batch(G, batch, deployed_vnfs,
                                     service_rate, K, T_func, T_proc,
                                     workload, remaining_capacity)
    all_assigns.update(assigns)

# # Visualization of final assignments 
# pos = nx.spring_layout(G, seed=42)
# for (s,d), info in all_assigns.items():
#     path = info['path']
#     edges = [(path[j],path[j+1]) for j in range(len(path)-1)]
#     nx.draw(G, pos, node_color='lightgray', edge_color='gray', with_labels=True)
#     nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='red', width=2.5)
#     plt.title(f"{s}→{d}, delay={info['delay']:.1f}")
#     plt.show()
