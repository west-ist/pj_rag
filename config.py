#配置管理
import os
from pathlib import Path
#辅助函数：读取密钥(LLM辅助）
def load_env():
    env_file = Path(__file__).parent / '.env'
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    # 去除可能的引号
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    if value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    os.environ[key] = value#设置环境变量为密钥，后面通过getenv函数就可以读取这个环境变量

load_env()

#设置必要的变量
Dashscope_key = os.getenv("Dashscope_key","")
model = "qwen-plus"
embedding_model = "text-embedding-v1"
Top_k = 3
random_seed = 42
##路径
project_root = Path(__file__).parent
knowledge_root = project_root/"mock_data"/"knowledge_base.json"
QA_root = project_root/"mock_data"/"qa_testset.csv"
evaluation_root = project_root/"outputs"/"qa_eval.csv"
failure_root = project_root/"outputs"/"qa_failure.md"
cache_file = project_root/"outputs"/"embeddings_cache.pkl"#embedding缓存文件路径

#检查
if not Dashscope_key:
    print("请正确设置API密钥\n")