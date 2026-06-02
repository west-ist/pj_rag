#rag主流程
import config
from data_loader import load_knowledge_base
from embedding_retriever import VectorRetriever
from llm_client import call_llm

retriever = None
initialized = False
##初始化
def init_retriever():
    global retriever, initialized
    if initialized:
        return
    texts,metadata = load_knowledge_base(config.knowledge_root)
    retriever = VectorRetriever(texts,metadata)
    print("RAG系统已成功初始化\n")
    initialized = True

#RAG回答：把最相关的top-k个文档作为上下文，调用llm生成回答
def rag_answer(question:str)->str:
    global retriever
    if retriever is None:
        init_retriever()

    retrieved = retriever.retrieve(question,k = config.Top_k)
    context = []
    for i,j in enumerate(retrieved,1):
        meta = j["metadata"]
        source = ""
        if meta.get("source_title"):
            source = f"(来源：{meta['source_title']}"
            if meta.get("article_ref"):
                source += f" {meta['article_ref']}"
                if meta.get("chapter"):
                    source += f" {meta['chapter']}"
            source += ")"
        context.append(f"【参考片段 {i}】(相似度: {j['similarity']:.3f}) {source}\n{j['text']}")
    contexts = "\n\n".join(context)
    prompt = f"""请根据以下参考资料回答问题，要求回答简洁，抓住重点。
    {contexts}

    问题：{question}

    回答："""
    return call_llm(prompt)

#与无rag的回答对比
def non_rag_answer(question:str)->str:
    prompt = f"请简洁回答以下法律问题：{question}"
    return call_llm(prompt,system_message="你是一个专业的法律助手")
#二编：修改了两种回答的prompt，避免输出过长回答
#返回检索到的知识及其相似度，用于评估
def get_retrieve(question:str)->str:
    global retriever
    if retriever is None:
        init_retriever()
    return retriever.retrieve(question,k = config.Top_k)