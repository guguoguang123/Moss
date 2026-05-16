"""
MOSS 论文复现：网表到有向图 (Directed Graph) 解析器
核心任务：将 Yosys 生成的门级/RTL混合网表，转化为深度学习框架 (GNN) 可读的拓扑图。
关键处理：清洗 Yosys 注释、提取时序锚点 (DFF)、解析组合逻辑 (包含 MUX 门)、控制出图数量。
"""

import os
import re
import pickle
import networkx as nx
import matplotlib.pyplot as plt

class MOSSDualModalGraphParser:
    def __init__(self, netlist_dir, output_dir):
        self.netlist_dir = netlist_dir
        self.output_dir = output_dir
        
        # ==================== 核心正则表达式配置区 ====================
        # 1. 匹配组合逻辑门：抓取 assign 语句
        self.assign_pattern = re.compile(r"assign\s+([^=]+)\s*=\s*(.+?);")
        
        # 2. 匹配门级触发器：抓取 Yosys 映射后的底层原语
        self.dff_pattern = re.compile(r"\$_\w*DFF\w*_\s+([^ ]+)\s*\((.*?)\);")
        
        # 3. 匹配寄存器声明：为了保留 RTL 变量名以供 LLM 对齐
        self.reg_pattern = re.compile(r"^\s*reg\s+(?:\[\d+:\d+\]\s+)?([a-zA-Z_0-9]+);")
        
        # 4. 匹配时序数据流入：抓取 always 块内的非阻塞赋值
        self.dff_assign_pattern = re.compile(r"([a-zA-Z_0-9]+)(?:\[\d+\])?\s*<=\s*([a-zA-Z_0-9_]+)")
        # =========================================================

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def clean_yosys_line(self, line):
        """
        [数据清洗] 
        擦除 Yosys 网表中混杂的 (* src = ... *) 和 /* ... */ 注释，防止污染节点名。
        """
        line = re.sub(r'\(\*.*?\*\)', '', line) 
        line = re.sub(r'/\*.*?\*/', '', line)   
        return line.strip()

    def parse_single_netlist(self, filepath):
        """解析单个 .v 网表文件，返回一个 NetworkX 有向图对象"""
        G = nx.DiGraph()
        module_name = os.path.basename(filepath).replace("_netlist.v", "")
        
        with open(filepath, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        gate_counter = 0 # 用于为组合逻辑门分配唯一的匿名 ID

        for line in lines:
            line = self.clean_yosys_line(line)
            if not line:
                continue
            
            # ==============================================================
            # 阶段一：提取时序逻辑节点 (DFF) —— 这是未来 LLM 注入特征的“锚点”
            # ==============================================================
            
            # 分支 A：处理标准的门级 DFF 原语
            dff_match = self.dff_pattern.search(line)
            if dff_match:
                dff_name = dff_match.group(1).strip('\\').strip()
                pins = dff_match.group(2)
                G.add_node(dff_name, node_type="DFF", llm_embedding=None)
                
                d_match = re.search(r"\.D\((.*?)\)", pins)
                q_match = re.search(r"\.Q\((.*?)\)", pins)
                if d_match:
                    G.add_edge(d_match.group(1).strip(), dff_name, pin="D")
                if q_match:
                    G.add_edge(dff_name, q_match.group(1).strip(), pin="Q")
                continue
                
            # 分支 B：处理被保留下来的 RTL 风格寄存器声明
            reg_match = self.reg_pattern.search(line)
            if reg_match:
                reg_name = reg_match.group(1).strip()
                G.add_node(reg_name, node_type="DFF", llm_embedding=None)
                continue
                
            # 分支 C：通过 `<=` 锁定数据流向 DFF 的通路
            if "<=" in line:
                assign_match = self.dff_assign_pattern.search(line)
                if assign_match:
                    target_reg = assign_match.group(1).strip()
                    source_wire = assign_match.group(2).strip()
                    # 排除常数赋值
                    if "'" not in source_wire and not source_wire.isdigit():
                        if not G.has_node(target_reg):
                            G.add_node(target_reg, node_type="DFF", llm_embedding=None)
                        G.add_edge(source_wire, target_reg, pin="D")
                continue
                
            # ==============================================================
            # 阶段二：提取组合逻辑节点 (AND/OR/NOT/XOR/MUX)
            # ==============================================================
            assign_match = self.assign_pattern.search(line)
            if assign_match:
                target_wire = assign_match.group(1).strip().split('[')[0].strip()
                expr = assign_match.group(2).strip()
                
                gate_node = f"{module_name}_gate_{gate_counter}"
                gate_counter += 1
                
                # 1. 非门 (NOT)
                if expr.startswith("~"):
                    src = expr.replace("~", "").strip().split('[')[0].strip()
                    G.add_node(gate_node, node_type="NOT_GATE")
                    G.add_edge(src, gate_node)
                    G.add_edge(gate_node, target_wire)
                # 2. 与门 (AND)
                elif "&" in expr:
                    srcs = [s.strip().split('[')[0].strip() for s in expr.split("&")]
                    G.add_node(gate_node, node_type="AND_GATE")
                    for src in srcs: G.add_edge(src, gate_node)
                    G.add_edge(gate_node, target_wire)
                # 3. 或门 (OR)
                elif "|" in expr:
                    srcs = [s.strip().split('[')[0].strip() for s in expr.split("|")]
                    G.add_node(gate_node, node_type="OR_GATE")
                    for src in srcs: G.add_edge(src, gate_node)
                    G.add_edge(gate_node, target_wire)
                # 4. 异或门 (XOR)
                elif "^" in expr:
                    srcs = [s.strip().split('[')[0].strip() for s in expr.split("^")]
                    G.add_node(gate_node, node_type="XOR_GATE")
                    for src in srcs: G.add_edge(src, gate_node)
                    G.add_edge(gate_node, target_wire)
                # 5. 多路复用器 (MUX)
                elif "?" in expr and ":" in expr:
                    G.add_node(gate_node, node_type="MUX_GATE")
                    cond = expr.split("?")[0].strip().split('[')[0].strip()
                    rest = expr.split("?")[1]
                    t_val = rest.split(":")[0].strip().split('[')[0].strip()
                    f_val = rest.split(":")[1].strip().split('[')[0].strip()
                    
                    if "'" not in cond and not cond.isdigit(): G.add_edge(cond, gate_node)
                    if "'" not in t_val and not t_val.isdigit(): G.add_edge(t_val, gate_node)
                    if "'" not in f_val and not f_val.isdigit(): G.add_edge(f_val, gate_node)
                    G.add_edge(gate_node, target_wire)

        return G, module_name

    def process_dataset(self, max_visualize_nodes=300, max_visualize_count=3):
        """
        处理整个数据集文件夹。
        :param max_visualize_nodes: 超过此节点数的电路不画图，防内存溢出。
        :param max_visualize_count: 全局最多生成多少张预览图片。
        """
        print(f"[*] 启动 MOSS 图结构解析器 (全库扫描模式)...")
        graph_collection = {}
        dff_count_total = 0
        processed_count = 0
        drawn_images_count = 0 # 记录已生成的图片数量

        for root, dirs, files in os.walk(self.netlist_dir):
            for file in files:
                if file.endswith(".v"):
                    file_path = os.path.join(root, file)
                    G, mod_name = self.parse_single_netlist(file_path)
                    
                    if G.number_of_nodes() > 0:
                        dff_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'DFF']
                        dff_count_total += len(dff_nodes)
                        graph_collection[mod_name] = G
                        processed_count += 1
                        print(f"  -> [解析成功] {mod_name:<15} | 节点数: {G.number_of_nodes():<5} | 边数: {G.number_of_edges():<5} | DFF锚点: {len(dff_nodes)}")
                        
                        # 【双重安全限制】：节点少于指定值，且总画图数未达标才画图
                        if G.number_of_nodes() <= max_visualize_nodes and drawn_images_count < max_visualize_count:
                            self.visualize_graph(G, mod_name)
                            drawn_images_count += 1

        print("-" * 50)
        print(f"[+] 物理拓扑构建完成！全库共成功解析 {processed_count} 个电路模块。")
        print(f"[+] 累计捕获 {dff_count_total} 个触发器 (DFF) 锚点作为大模型特征接口。")
        print(f"[+] 预览图生成完毕，共为您生成了 {drawn_images_count} 张拓扑可视化图片。")
        return graph_collection

    def visualize_graph(self, G, name):
        """将解析后的有向图保存为可视化图片"""
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G, k=0.2, iterations=20) 
        
        node_colors = ['#FF6666' if d.get('node_type') == 'DFF' else '#99FF99' if str(d.get('node_type')).endswith('GATE') else '#DDDDDD' for n, d in G.nodes(data=True)]
        node_sizes = [400 if d.get('node_type') == 'DFF' else 200 if str(d.get('node_type')).endswith('GATE') else 50 for n, d in G.nodes(data=True)]
        
        nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=node_sizes, font_size=5, edge_color='gray', arrows=True, arrowsize=6)
        plt.title(f"MOSS Graph Preview: {name}")
        
        save_path = os.path.join(self.output_dir, f"{name}_moss_graph.png")
        plt.savefig(save_path, dpi=200)
        plt.close()

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    NETLIST_INPUT_DIR = os.path.join(current_dir, "../../data/NetList/netlist")
    GRAPH_OUTPUT_DIR = os.path.join(current_dir, "../../data/DataSet/Graphs")
    
    # 实例化解析器
    parser = MOSSDualModalGraphParser(netlist_dir=NETLIST_INPUT_DIR, output_dir=GRAPH_OUTPUT_DIR)
    
    # 开始解析：过滤节点>300的大电路，且全局最多只输出 3 张图片
    all_graphs = parser.process_dataset(max_visualize_nodes=300, max_visualize_count=3)
    
    # 序列化保存至本地二进制文件
    output_pkl = os.path.join(GRAPH_OUTPUT_DIR, "moss_graph_dataset.pkl")
    with open(output_pkl, 'wb') as f:
        pickle.dump(all_graphs, f)
        
    print(f"\n[数据持久化成功] 纯净无污染的二进制数据集已写入: {output_pkl}")