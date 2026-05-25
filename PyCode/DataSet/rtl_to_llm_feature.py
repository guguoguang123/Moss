# 导入操作系统相关的标准库，用于文件路径拼接、目录遍历等操作
import os
# 设置 HuggingFace 镜像端点，指向国内 HF 极速镜像站，加速模型下载（必须在导入 transformers 前设置才生效）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 导入 pickle 序列化库，用于读写 Python 对象的二进制文件（.pkl）
import pickle
# 导入正则表达式库，用于在 Verilog 源码中按寄存器名匹配代码行
import re
# 导入 PyTorch 深度学习框架，提供 GPU 张量计算与自动微分能力
import torch
# 从 transformers 库导入自动分词器和因果语言模型加载器
from transformers import AutoTokenizer, AutoModelForCausalLM

# 定义 MOSS 项目的大模型特征注入器类，负责将 Yi-Coder 的语义 embedding 注入到网表图的 DFF 节点中
class MOSSYiCoderFeatureInjector:
    # 构造函数：初始化输入 pkl 路径、RTL 源码目录、输出 pkl 路径
    def __init__(self, pkl_path, rtl_dir, output_pkl_path):
        # 保存输入图数据集的 .pkl 文件路径
        self.pkl_path = pkl_path
        # 保存 RTL 源码目录路径，用于后面按模块名查找 .sv/.v 文件
        self.rtl_dir = rtl_dir
        # 保存最终融合数据集输出路径
        self.output_pkl_path = output_pkl_path
        # 自动检测可用设备：有 CUDA GPU 就用 'cuda'，否则用 'cpu'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 输出提示信息，告知用户正在从 HuggingFace 加载模型
        print("[*] 正在从 HuggingFace 加载 Yi-Coder-9B-Chat (请耐心等待下载)...")
        # 指定 HuggingFace 上的模型 ID：01-ai 组织的 Yi-Coder-9B-Chat
        model_id = "01-ai/Yi-Coder-9B-Chat"
        
        # 加载分词器（tokenizer），trust_remote_code=True 允许模型执行自定义代码以适配特殊 token
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        # 以 bfloat16 精度加载大模型，将显存占用压缩到约 18GB，防止消费级 GPU OOM（显存溢出）
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,                        # 模型标识
            torch_dtype=torch.bfloat16,      # 使用 bf16 半精度浮点，兼顾数值范围与显存效率
            device_map="auto",               # 自动将模型各层分配到 GPU/CPU，支持多卡或单卡+CPU offload
            output_hidden_states=True,       # 强制模型返回所有层的隐藏状态，用于提取中间层 embedding
            trust_remote_code=True           # 信任模型仓库中的自定义代码
        )
        # 将模型切换到 eval（评估）模式，关闭 Dropout 等训练专用层，确保推理时输出稳定
        self.model.eval()
        # 从模型配置中读取隐藏层维度大小（Yi-Coder-9B 为 4096）
        self.hidden_dim = self.model.config.hidden_size
        # 输出模型加载成功信息，并打印特征维度供确认
        print(f"[+] 大模型加载成功！特征维度锁定为: {self.hidden_dim}")

    # 加载网表图数据集（NetworkX 图字典），从 .pkl 文件反序列化
    def load_dataset(self):
        # 以二进制只读模式打开 pkl 文件
        with open(self.pkl_path, 'rb') as f:
            # 用 pickle 反序列化加载整个数据集并返回
            return pickle.load(f)

    # 在 RTL 源码目录中查找指定模块名的 SystemVerilog/Verilog 源文件
    def find_rtl_file(self, module_name):
        # 递归遍历 RTL 目录树，获取所有文件和子目录
        for root, dirs, files in os.walk(self.rtl_dir):
            # 遍历当前目录下的每个文件
            for file in files:
                # 不区分大小写匹配：module_name.sv 或 module_name.v
                if file.lower() in [f"{module_name.lower()}.sv", f"{module_name.lower()}.v"]:
                    # 找到后返回该文件的完整绝对路径
                    return os.path.join(root, file)
        # 遍历完未找到则返回 None，表示此模块没有对应的 RTL 源码
        return None

    # 从 RTL 源文件中提取包含目标寄存器的上下文代码行（代码切片）
    def extract_context(self, sv_path, target_reg, module_name):
        # 如果路径为空或文件不存在，返回一个简短的文本描述作为兜底
        if not sv_path or not os.path.exists(sv_path):
            return f"Register {target_reg} in {module_name}."
            
        # 初始化一行行匹配到的代码行列表
        context_lines = []
        # 以 UTF-8 编码打开 RTL 源文件，errors='ignore' 跳过无法解码的字符
        with open(sv_path, 'r', encoding='utf-8', errors='ignore') as f:
            # 逐行读取文件内容
            for line in f:
                # 使用正则按单词边界精确匹配目标寄存器名（避免子串误匹配）
                if re.search(rf'\b{target_reg}\b', line):
                    # 去除行首尾空白后加入上下文列表
                    context_lines.append(line.strip())
                    
        # 如果至少匹配到了一行，拼成自然语言风格的描述返回
        if context_lines:
            return f"In Verilog design {module_name}, the register logic for '{target_reg}' is: {' '.join(context_lines)}"
        # 未匹配到任何行则返回简短描述
        return f"Register component named {target_reg} in {module_name}."

    # 调用 Yi-Coder 大模型将代码文本转化为高维语义向量（embedding）
    def get_yi_coder_embedding(self, text):
        # 使用分词器将文本转为 token id 张量，限制最大长度为 512，并移到指定设备
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        # 禁用梯度计算，加速推理并节省显存
        with torch.no_grad():
            # 将 token 输入模型，拿到包含 hidden_states 的输出
            outputs = self.model(**inputs)
            # 取最后一层 Transformer 的隐藏状态 (shape: [batch, seq_len, hidden_dim])
            last_layer_hidden = outputs.hidden_states[-1]
            # 对序列维度做 Mean-Pooling，将变长序列压缩为固定维度向量，再转 float32 并转到 CPU 转为 NumPy 数组
            embedding = last_layer_hidden.mean(dim=1).squeeze().float().cpu().numpy()
        # 返回最终的 NumPy 特征向量
        return embedding

    # 核心流程：遍历整个图数据集，为 DFF 节点注入大模型特征，为组合逻辑节点填充零向量
    def run_injection(self):
        # 加载原始的 NetworkX 图字典数据集
        dataset = self.load_dataset()
        # 计数器：统计总共为多少 DFF 节点注入了真实 embedding
        total_injected = 0
        
        # 输出开始注入的提示
        print("[*] 开始阅读 RTL 源码并注入特征...")
        # 遍历数据集中的每个模块（键为模块名，值为 NetworkX 图对象）
        for module_name, G in dataset.items():
            # 查找该模块对应的 RTL 源文件路径
            sv_path = self.find_rtl_file(module_name)
            # 列表推导式：筛选图中 node_type 属性为 'DFF' 的所有节点
            dff_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'DFF']
            
            # 步骤 1：为每个 DFF 节点生成并注入大模型语义特征向量
            for dff in dff_nodes:
                # 从 RTL 源码中提取该寄存器的上下文描述文本
                context_text = self.extract_context(sv_path, dff, module_name)
                # 调用大模型将上下文文本转为高维 embedding
                feature_vector = self.get_yi_coder_embedding(context_text)
                # 将 embedding 写入图节点的 'llm_embedding' 属性
                G.nodes[dff]['llm_embedding'] = feature_vector
                # 注入计数器 +1
                total_injected += 1
                
            # 步骤 2：为所有非 DFF 节点（组合逻辑门等）填充全零向量，保证所有节点特征维度一致
            for n, d in G.nodes(data=True):
                # 如果节点类型不是 DFF（即组合逻辑门、输入输出端口等）
                if d.get('node_type') != 'DFF':
                    # 用全零向量填充，维度与 hidden_dim 对齐
                    G.nodes[n]['llm_embedding'] = [0.0] * self.hidden_dim
                    
            # 输出当前模块的注入统计信息（模块名左对齐占 15 个字符宽度）
            print(f"  -> [成功] 模块 {module_name:<15} | 注入了 {len(dff_nodes)} 个节点的特征")

        # 将融合后的数据集以二进制形式写入输出 pkl 文件
        with open(self.output_pkl_path, 'wb') as f:
            pickle.dump(dataset, f)
        # 打印分隔线
        print("-" * 50)
        # 输出最终完成提示和输出文件路径
        print(f"[+] 终极多模态数据集已生成！保存至: {self.output_pkl_path}")

# 脚本入口：当直接运行此文件时（而非被 import），执行以下代码
if __name__ == "__main__":
    # 获取当前脚本所在的目录绝对路径，作为基准路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 拼接出 RTL 源码目录的路径（向上两级再进入 data/RTL）
    RTL_DIR = os.path.join(current_dir, "../../data/RTL")
    # 拼接出输入图数据集的 .pkl 文件路径
    INPUT_PKL = os.path.join(current_dir, "../../data/DataSet/Graphs/moss_graph_dataset.pkl")
    # 拼接出输出融合数据集的 .pkl 文件路径
    OUTPUT_PKL = os.path.join(current_dir, "../../data/DataSet/Graphs/moss_fused_dataset.pkl")
    
    # 实例化注入器对象
    injector = MOSSYiCoderFeatureInjector(INPUT_PKL, RTL_DIR, OUTPUT_PKL)
    # 启动特征注入流程
    injector.run_injection()