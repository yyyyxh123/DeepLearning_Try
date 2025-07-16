import networkx as nx
import matplotlib.pyplot as plt

# 创建一个有向图
G = nx.DiGraph()

# 添加节点和边
G.add_edges_from([
    ('A', 'B'),
    ('B', 'C'),
    ('A', 'D'),
    ('D', 'C')
])

# 判断是否为 DAG
print("Is DAG:", nx.is_directed_acyclic_graph(G))

# 可视化
nx.draw(G, with_labels=True, node_color='lightblue', arrows=True)
plt.title("Example of a DAG")
plt.show()
