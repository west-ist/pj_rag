#加载数据
import csv
import json
from pathlib import Path
from typing import List, Dict, Tuple
import config
##加载知识库文件，每个内容作为一个chunk
def load_knowledge_base(json_file:Path)->Tuple[List[str],List[Dict]]:
    with open(json_file,'r',encoding='utf-8-sig') as f:
        data = json.load(f)
    texts = []
    metadata = []
    for item in data:
        text = item.get("text","").strip()
        if not text:
            continue
        texts.append(text)
        metadata.append(item.get("metadata",{}))
    print(f"成功加载知识库，共{len(texts)}个chunk\n")
    return texts, metadata
##加载QA测试集
def load_qa_testset(directory:Path)->List[Dict[str, str]]:
    qalist = []
    with open(directory, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row["question"].strip()
            a = row["answer"].strip()
            qalist.append({"question": q, "answer": a})
    print(f"成功加载数据集，共{len(qalist)}个问题\n")
    return qalist
