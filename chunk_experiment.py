#二编：独立的三种大小的chunk对比，分别生成对应的csv文件，手动标注correct后计算正确率，生成对比柱状图
import csv
import time
import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import config
from data_loader import load_qa_testset
from embedding_retriever import VectorRetriever, get_embedding
from llm_client import call_llm
import dashscope

dashscope.api_key = config.Dashscope_key

def rag_answer_retriever(question: str, retriever, k: int = config.Top_k) -> str:
    retrieved = retriever.retrieve(question, k=k)
    context_parts = []
    for i, item in enumerate(retrieved, 1):
        meta = item.get("metadata", {})
        source = ""
        if meta.get("source_title"):
            source = f"（来源：{meta['source_title']}"
            if meta.get("article_ref"):
                source += f" {meta['article_ref']}"
            if meta.get("chapter"):
                source += f" {meta['chapter']}"
            source += "）"
        context_parts.append(
            f"【参考片段 {i}】(相似度: {item['similarity']:.3f}) {source}\n{item['text']}"
        )
    contexts = "\n\n".join(context_parts)
    prompt = f"""请根据以下参考资料回答问题，要求回答简洁，抓住重点。
{contexts}

问题：{question}

回答："""
    return call_llm(prompt)

def eval_one_chunk(kb_json_path: Path, output_csv: Path, chunk_size: int):
    print(f"正在评估知识库 {kb_json_path.name}\n")
    with open(kb_json_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    texts = []
    metadata = []
    for item in data:
        text = item.get("text", "").strip()
        if not text:
            continue
        texts.append(text)
        metadata.append(item.get("metadata", {}))
    print(f"成功加载知识库，共{len(texts)}个chunk\n")

    # 设置当前 chunk 大小的独立缓存路径
    config.cache_file = config.get_cache_path(chunk_size)
    retriever = VectorRetriever(texts, metadata)

    qa_list = load_qa_testset(config.QA_root)
    output_csv.parent.mkdir(exist_ok=True, parents=True)

    with output_csv.open("w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question_id", "question", "answer", "rag_answer", "correct", "retrieval_scores"])
        for idx, item in enumerate(tqdm(qa_list, desc="评估进度"), 1):
            q = item["question"]
            a = item["answer"]
            rag_ans = rag_answer_retriever(q, retriever)
            time.sleep(0.5)
            retrieved = retriever.retrieve(q, k=config.Top_k)
            scores = [round(float(doc["similarity"]), 4) for doc in retrieved]
            writer.writerow([idx, q, a, rag_ans, "", str(scores)])
    print("评估完成\n")

def count_accuracy(csv_path: Path) -> float:
    df = pd.read_csv(csv_path, encoding="utf-8")
    if 'correct' not in df.columns:
        raise ValueError("缺少 'correct' 列，请先人工标注。")
    rag_acc = pd.to_numeric(df['correct'], errors='coerce').mean() * 100
    return rag_acc

def draw_comparison(acc_dict: dict, save_path: Path):
    sizes = sorted(acc_dict.keys())
    acc = [acc_dict[s] for s in sizes]
    x_pos = range(len(sizes))
    plt.bar(x_pos, acc, width=0.6, color='steelblue')
    plt.xticks(x_pos, sizes)
    plt.xlabel('chunk size(characters)')
    plt.ylabel('accuracy (%)')
    plt.title('RAG accuracy by chunk size')
    plt.savefig(save_path)
    plt.show()

def main():
    if not config.chunk_experiment:
        print("请在config中将chunk_experiment设置为True\n")
        return
    experiment_dir = config.chunk_experiment_output
    experiment_dir.mkdir(exist_ok=True, parents=True)
    csv_files = {}
    for size, kb_path in config.chunk_sizes.items():
        output_csv = experiment_dir / f"qa_eval_chunk_{size}.csv"
        eval_one_chunk(kb_path, output_csv, chunk_size=size)
        csv_files[size] = output_csv

    print("请手动标注 correct 列，之后按回车生成对比图\n")
    input()

    accuracies = {}
    for size, path in csv_files.items():
        try:
            acc = count_accuracy(path)
            accuracies[size] = acc
        except Exception as e:
            print(f"读取 {path} 失败: {e}")
            continue
    if not accuracies:
        print("没有有效的正确率数据，请检查标注是否正确。\n")
        return

    plot_path = experiment_dir / "chunk_accuracy_comparison.png"
    draw_comparison(accuracies, plot_path)

if __name__ == "__main__":
    main()