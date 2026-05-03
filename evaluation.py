import csv
import time
from tqdm import tqdm
from rag_pipeline import init_retriever, rag_answer, non_rag_answer,get_retrieve
from data_loader import load_qa_testset
import config
#评估脚本。包含问题、实际答案、rag答案、非rag答案、hit@k初判（用于判断答案是否命中了给定的知识）、人工判断正确率、retrieval score(余弦相似度）
def run_evaluation():
    print("开始评估\n")
    init_retriever()
    qa_list = load_qa_testset(config.QA_root)
    config.evaluation_root.parent.mkdir(parents=True, exist_ok=True)
    with open(config.evaluation_root, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "question_id", "question", "answer",
            "non_rag_answer", "rag_answer",
            "hit@k", "correct", "retrieval_scores"
        ])
        for idx, i in enumerate(tqdm(qa_list, desc="评估进度"), 1):
            q = i["question"]
            a = i["answer"]
            #获取回答（加入适当延时避免限流）
            nonrag_ans = non_rag_answer(q)
            time.sleep(0.5)
            rag_ans = rag_answer(q)
            time.sleep(0.5)
            # 检索信息
            retrieved = get_retrieve(q)
            hit = _compute_hit(retrieved, a)
            scores = [round(float(i["similarity"]), 4) for i in retrieved]
            writer.writerow([
                idx, q, a,
                nonrag_ans, rag_ans,
                hit, "",   #correct等待人工标注
                str(scores)
            ])

    print("评估完成\n")
    print("请手动判断正确率，正确为1，错误为0。并运行failure_analyzer.py生成失败案例报告\n")

#辅助函数，计算hit@k（该概念由llm解释）
def _compute_hit(retrieved, answer: str)->str:
    #提取前20个非停用词作为答案的关键词，如回答命中至少一个，则标为命中
    keywords = set(answer[:100].split())
    for i in retrieved:
        text = i["text"]
        if any(j in text for j in keywords):
            return "1"
    return "0"