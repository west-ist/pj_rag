import numpy as np
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import traceback
from pathlib import Path
import config
from dashscope import TextEmbedding
#向量检索

#建立类
class VectorRetriever:
    def __init__(self, texts: List[str],metadata: List[Dict], use_cache: bool = True):
        self.texts = texts
        self.metadata = metadata
        self.use_cache = use_cache
        self.embeddings = None
        self._build_index()
##保存文档列表，如果缓存中已存在，直接加载，否则计算embedding，保存到缓存（LLM提供思路）
    def _build_index(self):
        cache_path = config.cache_file
        if self.use_cache and cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    cache = pickle.load(f)
                    self.embeddings = cache["embeddings"]
                    print("成功从缓存加载embedding\n")
                    return
            except Exception as e:
                print("无法从缓存加载，重新计算\n")

        self.embeddings = []
        total =len(self.texts)
        for text in self.texts:
            self.embeddings.append(get_embedding(text))
        self.embeddings = np.array(self.embeddings)
        if self.use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'texts': self.texts,
                    'embeddings': self.embeddings
                }, f)
            print(f"已保存embedding缓存到{cache_path}\n")
##检索最相似的k个文档片段
    def retrieve(self,query:str,k:int = config.Top_k)->List[Dict]:
        q_embedding = get_embedding(query)
        similarity = cosine_similarity([q_embedding], self.embeddings)[0]
        top_k = np.argsort(similarity)[::-1][:k]
        results = []
        for index in top_k:
            results.append({"text":self.texts[index],"similarity":float(similarity[index]),"metadata":self.metadata[index]})
        return results
##调用API，将文本转化为向量（代码经过LLM修改，解决API无法正确调用问题）
def get_embedding(text:str)->np.ndarray:
    if not config.Dashscope_key or config.Dashscope_key == "":
        raise RuntimeError("API Key 未配置")

    resp = TextEmbedding.call(
        model=config.embedding_model,
        input=text,
    )
    if resp.get('status_code') != 200:
        raise RuntimeError(f"Embedding API 错误: {resp.get('code')} - {resp.get('message')}")

    emb = resp['output']['embeddings'][0]['embedding']
    return np.array(emb, dtype=np.float32)


