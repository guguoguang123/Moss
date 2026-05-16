import os
import re
import pickle
import time
import networkx as nx
import matplotlib.pyplot as plt

class MOSSDualModalGraphParser:
    """
    严格基于 MOSS 论文架构的网表图构建器 (Cell-centric Graph)
    核心功能：提取组合逻辑门、时序逻辑节点(DFF/reg)，并为 LLM 语义特征注入预留锚点。
    """
    def __init__(self, netlist_dir, output_dir):
        self.netlist_dir = netlist_dir
        self.output_dir = output_dir
        
        # ==================== 正则表达式配置区 ====================
        
        # 1. 匹配 Yosys 组合逻辑门 (例如: assign _00_ = ~a; 或 assign _01_ = a | b;)
        self.assign_pattern = re.compile(r"assign\s+([^=]+)\s*=\s*(.+?);")
        
        # 2. 匹配标准的门级触发器 (如果 simplemap 成功完全映射)
        self.dff_pattern = re.compile(r"\$_\w*DFF\w*_\s+([^ ]+)\s*\((.*?)\);")

        # 3. 【兼容模式】匹配 Yosys 未打碎的寄存器声明 (例如: reg [7:0] Q; 或 reg out;)
        self.reg_pattern = re.compile(r"^\s*reg\s+(?:\[\d+:\d+\]\s+)?([a-zA-Z_0-9]+);")
        
        # 4. 【兼容模式】匹配 always 块内的触发器数据流入 (例如: out <= _01_;)
        self.dff_assign_pattern = re.compile(r"([a-zA-Z_0-9]+)(?:\[\d+\])?\s*<=\s*([a-zA-Z_0-9_]+)")
        
        # =========================================================

        # 确保输出目录存在
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def parse_single_netlist(self, filepath):
        """解析单一网表文件，返回 NetworkX 有向图对象"""
        G = nx.DiGraph()
        module_name = os.path.basename(filepath).replace("_netlist.v", "")
        
        with open(filepath, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        gate_counter = 0 # 匿名逻辑门 ID 计数器

        for line in lines:
            line = line.strip()
            
            # ---------------------------------------------------------
            # 步骤 1: 提取时序节点 (DFF 锚点) - 抓门级原语
            # ---------------------------------------------------------
            dff_match = self.dff_pattern.search(line)
            if dff_match:
                dff_name = dff_match.group(2).strip('\\') # 去除转义斜杠
                pins = dff_match.group(3)
                
                G.add_node(dff_name, node_type="DFF", llm_embedding=None)
                
                d_match = re.search(r"\.D\((.*?)\)", pins)
                q_match = re.search(r"\.Q\((.*?)\)", pins)
                
                if d_match:
                    in_wire = d_match.group(1).strip()
                    G.add_edge(in_wire, dff_name, pin="D")
                if q_match:
                    out_wire = q_match.group(1).strip()
                    G.add_edge(dff_name, out_wire, pin="Q")
                continue
                
            # ---------------------------------------------------------
            # 步骤 2: 提取时序节点 (DFF 锚点) - 兼容 RTL 风格的 reg
            # ---------------------------------------------------------
            reg_match = self.reg_pattern.search(line)
            if reg_match:
                reg_name = reg_match.group(1).strip()
                G.add_node(reg_name, node_type="DFF", llm_embedding=None)
                continue
                
            # 抓取寄存器的数据流入 (如 <= )
            if "<=" in line:
                assign_match = self.dff_assign_pattern.search(line)
                if assign_match:
                    target_reg = assign_match.group(1).strip()
                    source_wire = assign_match.group(2).strip()
                    
                    # 排除常数复位赋值 (例如 1'h0, 1'b1)
                    if "'" not in source_wire:
                        if not G.has_node(target_reg):
                            G.add_node(target_reg, node_type="DFF", llm_embedding=None)
                        G.add_node(source_wire, type="signal")
                        G.add_edge(source_wire, target_reg, pin="D")
                continue
                
            # ---------------------------------------------------------
            # 步骤 3: 提取组合逻辑门节点
            # ---------------------------------------------------------
            assign_match = self.assign_pattern.search(line)
            if assign_match:
                target_wire = assign_match.group(1).strip()
                expr = assign_match.group(2).strip()
                
                gate_node = f"gate_{gate_counter}"
                gate_counter += 1
                
                if expr.startswith("~"):
                    src = expr.replace("~", "").strip()
                    G.add_node(gate_node, node_type="NOT_GATE")
                    G.add_edge(src, gate_node)
                    G.add_edge(gate_node, target_wire)
                elif "|" in expr:
                    srcs = [s.strip() for s in expr.split("|")]
                    G.add_node(gate_node, node_type="OR_GATE")
                    for src in srcs:
                        G.add_edge(src, gate_node)
                    G.add_edge(gate_node, target_wire)
                elif "&" in expr:
                    srcs = [s.strip() for s in expr.split("&")]
                    G.add_node(gate_node, node_type="AND_GATE")
                    for src in srcs:
                        G.add_edge(src, gate_node)
                    G.add_edge(gate_node, target_wire)
                elif "^" in expr:
                    srcs = [s.strip() for s in expr.split("^")]
                    G.add_node(gate_node, node_type="XOR_GATE")
                    for src in srcs:
                        G.add_edge(src, gate_node)
                    G.add_edge(gate_node, target_wire)

        return G, module_name

    def process_dataset(self, visualize_limit=2):
        print(f"[*] 启动 MOSS 双模态图提取器...")
        graph_collection = {}
        dff_count_total = 0
        processed_count = 0

        for root, dirs, files in os.walk(self.netlist_dir):
            for file in files:
                if file.endswith(".v"):
                    file_path = os.path.join(root, file)
                    G, mod_name = self.parse_single_netlist(file_path)
                    
                    if G.number_of_nodes() > 0:
                        # 统计提取到的 DFF 锚点数量
                        dff_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'DFF']
                        dff_count_total += len(dff_nodes)
                        
                        graph_collection[mod_name] = G
                        processed_count += 1
                        print(f"  -> [解析成功] {mod_name:<15} | 总节点: {G.number_of_nodes():<4} | DFF锚点: {len(dff_nodes)}")
                        
                        # 按限制数量生成可视化图片
                        if processed_count <= visualize_limit:
                            self.visualize_graph(G, mod_name)

        print("-" * 50)
        print(f"[+] 提取完成！共解析 {processed_count} 个电路模块。")
        print(f"[+] 全局共捕获 {dff_count_total} 个触发器 (DFF) 锚点。")
        return graph_collection

    def visualize_graph(self, G, name):
        """将有向图保存为直观的彩色节点图图片"""
        plt.figure(figsize=(12, 10))
        # 增加迭代次数防止节点重叠
        pos = nx.spring_layout(G, k=0.15, iterations=30) 
        
        node_colors = []
        node_sizes = []
        for n, d in G.nodes(data=True):
            if d.get('node_type') == 'DFF':
                node_colors.append('#FF6666') # 显眼红色：DFF 触发器
                node_sizes.append(600)
            elif str(d.get('node_type')).endswith('GATE'):
                node_colors.append('#99FF99') # 绿色：逻辑门
                node_sizes.append(300)
            else:
                node_colors.append('#DDDDDD') # 灰色：普通连线
                node_sizes.append(100)
                
        nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=node_sizes, 
                font_size=6, font_color='black', edge_color='gray', arrows=True, arrowsize=8)
        
        # 使用时间戳防止图片覆盖
        timestamp = int(time.time())
        save_path = os.path.join(self.output_dir, f"{name}_{timestamp}_moss.png")
        plt.title(f"MOSS Graph Topology: {name}\n(Red: DFF Anchors | Green: Logic Gates)")
        plt.savefig(save_path, dpi=200)
        plt.close()

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 动态获取路径，确保随处运行都不报错
    current_dir = os.path.dirname(os.path.abspath(__file__))
    NETLIST_INPUT_DIR = os.path.join(current_dir, "../../data/NetList/netlist")
    GRAPH_OUTPUT_DIR = os.path.join(current_dir, "../../data/DataSet/Graphs")
    
    # 1. 实例化并运行提取器
    parser = MOSSDualModalGraphParser(netlist_dir=NETLIST_INPUT_DIR, output_dir=GRAPH_OUTPUT_DIR)
    
    # visualize_limit 可以改为 0 (不画图省时间) 或者改大 (全画出来)
    all_graphs = parser.process_dataset(visualize_limit=5)
    
    # 2. 将真实的图结构数据(字典)保存为硬盘上的 pkl 文件
    dataset_file = os.path.join(GRAPH_OUTPUT_DIR, "moss_graph_dataset.pkl")
    
    # 采用二进制写入('wb')
    with open(dataset_file, 'wb') as f:
        pickle.dump(all_graphs, f)
        
    print(f"\n[序列化完毕] GNN 图结构数据集已打包保存至: {dataset_file}")
    print("右脑 (GNN Graph) 基建正式完工！随时可以喂给神经网络进行训练。")