import os
# 【新增】：强行接管下载路由，指向国内极速镜像站（必须放在最前面！）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import pickle
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class MOSSYiCoderFeatureInjector:
    def __init__(self, pkl_path, rtl_dir, output_pkl_path):
        self.pkl_path = pkl_path
        self.rtl_dir = rtl_dir
        self.output_pkl_path = output_pkl_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print("[*] 正在从 HuggingFace 加载 Yi-Coder-9B-Chat (请耐心等待下载)...")
        model_id = "01-ai/Yi-Coder-9B-Chat"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        # 使用 bf16 精度加载，将显存占用压缩到 18GB 左右，防止单卡 OOM
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            output_hidden_states=True, # 强迫模型吐出隐藏层特征
            trust_remote_code=True
        )
        self.model.eval()
        self.hidden_dim = self.model.config.hidden_size
        print(f"[+] 大模型加载成功！特征维度锁定为: {self.hidden_dim}")

    def load_dataset(self):
        with open(self.pkl_path, 'rb') as f:
            return pickle.load(f)

    def find_rtl_file(self, module_name):
        """扫描 data/RTL 目录寻找对应的源码"""
        for root, dirs, files in os.walk(self.rtl_dir):
            for file in files:
                if file.lower() in [f"{module_name.lower()}.sv", f"{module_name.lower()}.v"]:
                    return os.path.join(root, file)
        return None

    def extract_context(self, sv_path, target_reg, module_name):
        """代码切片：只提取包含该寄存器的代码行"""
        if not sv_path or not os.path.exists(sv_path):
            return f"Register {target_reg} in {module_name}."
            
        context_lines = []
        with open(sv_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if re.search(rf'\b{target_reg}\b', line):
                    context_lines.append(line.strip())
                    
        if context_lines:
            return f"In Verilog design {module_name}, the register logic for '{target_reg}' is: {' '.join(context_lines)}"
        return f"Register component named {target_reg} in {module_name}."

    def get_yi_coder_embedding(self, text):
        """将代码文本转化为 4096 维向量"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 获取最后一层隐藏状态，并进行 Mean-Pooling
            last_layer_hidden = outputs.hidden_states[-1]
            embedding = last_layer_hidden.mean(dim=1).squeeze().float().cpu().numpy()
        return embedding

    def run_injection(self):
        dataset = self.load_dataset()
        total_injected = 0
        
        print("[*] 开始阅读 RTL 源码并注入特征...")
        for module_name, G in dataset.items():
            sv_path = self.find_rtl_file(module_name)
            dff_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'DFF']
            
            # 1. 给 DFF 注入大模型真实特征
            for dff in dff_nodes:
                context_text = self.extract_context(sv_path, dff, module_name)
                feature_vector = self.get_yi_coder_embedding(context_text)
                G.nodes[dff]['llm_embedding'] = feature_vector
                total_injected += 1
                
            # 2. 给组合逻辑门填充 0 向量，保持矩阵维度对齐
            for n, d in G.nodes(data=True):
                if d.get('node_type') != 'DFF':
                    G.nodes[n]['llm_embedding'] = [0.0] * self.hidden_dim
                    
            print(f"  -> [成功] 模块 {module_name:<15} | 注入了 {len(dff_nodes)} 个节点的特征")

        with open(self.output_pkl_path, 'wb') as f:
            pickle.dump(dataset, f)
        print("-" * 50)
        print(f"[+] 终极多模态数据集已生成！保存至: {self.output_pkl_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    RTL_DIR = os.path.join(current_dir, "../../data/RTL")
    INPUT_PKL = os.path.join(current_dir, "../../data/DataSet/Graphs/moss_graph_dataset.pkl")
    OUTPUT_PKL = os.path.join(current_dir, "../../data/DataSet/Graphs/moss_fused_dataset.pkl")
    
    injector = MOSSYiCoderFeatureInjector(INPUT_PKL, RTL_DIR, OUTPUT_PKL)
    injector.run_injection()