import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from .options import Options
from .vectors import Sentence_Transformer
from .eval import *
from .utils import *
     
def inference(model1, model2, embedding_model, tokenizer1, tokenizer2, text, ex, decoding):
    pred = {}
    
    slope, intercept = fitting(model2, embedding_model, tokenizer2, text, decoding=decoding)
    text_similarity = calculateTextSimilarity(model1, embedding_model, tokenizer1, text, decoding=decoding, device=model1.device)
    all_prob_estimated = [i*slope + intercept for i in text_similarity]
    pred["PETAL"] = -np.mean(all_prob_estimated).item()
    
    ex["pred"] = pred
    return ex

def evaluate_data(original_data, model1, model2, embedding_model, tokenizer1, tokenizer2, decoding):
    print(f"all data size: {len(original_data)}")
    all_output = []
    for ex in tqdm(original_data): 
        text = ex["input"]
        new_ex = inference(model1, model2, embedding_model, tokenizer1, tokenizer2, text, ex, decoding)
        all_output.append(new_ex)
    return all_output

def run_petal_evaluation(target_model, surrogate_model, data, length, gpu_ids, 
                         embedding_model_name, decoding, output_dir):
    """
    封装了PETAL评估完整流程的接口函数。
    你可以从其他Python脚本中导入并调用此函数来执行一次完整的评估。

    Args:
        target_model (str): 目标模型的名称 (例如: 'pythia-6.9b')。
        surrogate_model (str): 代理模型的名称 (例如: 'gpt2-xl')。
        data (str): 数据集的名称 (例如: 'WikiMIA')。
        length (int): 处理文本的长度。
        gpu_ids (str): 使用的GPU ID (例如: '0')。
        embedding_model_name (str): 用于计算嵌入的句子转换器模型名称。
        decoding (str): 解码策略。
        output_dir (str): 保存结果的根目录。

    Returns:
        list: 包含所有样本及其预测结果的列表。
    """
    # 1. 配置环境变量
    # Configure environment variables
    print("--- 开始PETAL评估流程 ---")
    os.environ['HF_ENDPOINT'] = 'hf-mirror.com'
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    
    # 2. 创建输出目录
    # Create output directory
    full_output_dir = f"{output_dir}/{data}/{target_model}_{surrogate_model}"
    Path(full_output_dir).mkdir(parents=True, exist_ok=True)
    print(f"结果将保存到: {full_output_dir}")

    # 3. 加载数据集、模型和分词器
    # Load dataset, models, and tokenizers
    print("正在加载数据集...")
    original_data = prepare_dataset(data, length)
    
    print("正在加载目标模型和代理模型...")
    model1, model2, tokenizer1, tokenizer2 = load_model(target_model, surrogate_model)
    
    print("正在加载嵌入模型...")
    embedding_model = Sentence_Transformer(embedding_model_name, model1.device)

    # 4. 执行评估
    # Execute evaluation
    all_output = evaluate_data(original_data, model1, model2, embedding_model, tokenizer1, tokenizer2, decoding)
    
    # 5. 生成并保存结果图表
    # Generate and save the result figure
    print("正在生成结果图表...")
    fig_fpr_tpr(all_output, full_output_dir)
    
    print("--- 评估流程执行完毕 ---")
    return all_output


if __name__ == '__main__':
    args = Options()
    args = args.parser.parse_args()
    os.environ['HF_ENDPOINT'] = 'hf-mirror.com'
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    args.output_dir = f"{args.output_dir}/{args.data}/{args.target_model}_{args.surrogate_model}"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # load model and data
    original_data = prepare_dataset(args.data, args.length)
    model1, model2, tokenizer1, tokenizer2 = load_model(args.target_model, args.surrogate_model)

    # load embedding model
    embedding_model = Sentence_Transformer(args.embedding_model, model1.device)

    all_output = evaluate_data(original_data, model1, model2, embedding_model, tokenizer1, tokenizer2, args.decoding)
    fig_fpr_tpr(all_output, args.output_dir)

