# 导入必要的库
# import necessary libraries
import os
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Literal, List, Optional, Dict, Any, Union

# 从您的 run 脚本中导入核心评估函数
# Import the core evaluation function from your run.py script
from PETAL.run import run_petal_evaluation
# 导入TwoClass.py中的成员推断攻击函数
from BinaryClassification.TwoClass import run_membership_inference_attack

# ==============================================================================
# 1. 定义 API 的请求体模型
#    - 使用 Pydantic 定义请求参数, FastAPI 会自动进行数据校验
#    - Field 用于设置默认值和描述
# ==============================================================================
class SecurityEvaluationRequest(BaseModel):
    """
    安全评测请求的模型。
    """
    #
    # 必需参数
    #
    target_model: str = Field(
        ..., # '...' 表示这是一个必需字段
        description="需要进行成员推断攻击安全评测的目标模型名称, 例如: 'pythia-6.9b' 或 'facebook/opt-1.3b'。"
    )

    #
    # 可选参数 (带默认值)
    #
    surrogate_model: str = Field(
        default='gpt2-xl',
        description="用于辅助攻击的代理模型, 默认为 'gpt2-xl'。"
    )
    data: str = Field(
        default='WikiMIA',
        description="用于评测的数据集, 默认为 'WikiMIA'。"
    )
    embedding_model: str = Field(
        default='bge-large-en-v1.5',
        description="用于生成文本嵌入的句向量模型, 默认为 'bge-large-en-v1.5'。"
    )
    length: int = Field(
        default=32,
        description="评测时使用的文本长度, 默认为 32。"
    )

    # 声明一个符合 OpenAPI 规范的例子
    # Declare an example compliant with the OpenAPI specification
    class Config:
        json_schema_extra = {
            "example": {
                "target_model": "pythia-6.9b",
                "surrogate_model": "gpt2-xl",
                "data": "WikiMIA",
                "embedding_model": "bge-large-en-v1.5",
                "length": 64
            }
        }

# 为TwoClass.py的成员推断攻击定义请求模型
class ClassicMIARequest(BaseModel):
    """
    经典成员推断攻击请求的模型。
    基于影子模型和二分类器的攻击方法。
    """
    # 必需参数
    target_model_name: str = Field(
        ...,
        description="目标模型名称, 例如: 'resnet18', 'lenet'。"
    )

    # 可选参数 (带默认值)
    dataset_name: str = Field(
        default='CIFAR10',
        description="数据集名称, 目前支持 'CIFAR10'。"
    )
    data_root: str = Field(
        default='./data',
        description="数据集存储路径。"
    )
    weights_dir: str = Field(
        default='./mia_weights',
        description="模型权重保存目录。"
    )
    target_train_size: int = Field(
        default=15000,
        description="目标模型训练集大小。"
    )
    n_shadow_models: int = Field(
        default=4,
        description="影子模型数量。"
    )
    target_epochs: int = Field(
        default=50,
        description="目标模型训练轮数。"
    )
    attack_epochs: int = Field(
        default=50,
        description="攻击模型训练轮数。"
    )
    force_retrain: bool = Field(
        default=False,
        description="是否强制重新训练模型(即使存在已保存的权重)。"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "target_model_name": "resnet18",
                "dataset_name": "CIFAR10",
                "n_shadow_models": 4,
                "target_train_size": 15000,
                "target_epochs": 50,
                "attack_epochs": 50,
                "force_retrain": False
            }
        }

# ==============================================================================
# 2. 创建 FastAPI 应用实例
# ==============================================================================
app = FastAPI(
    title="模型安全评测 API (成员推断攻击)",
    description="本 API 用于对大语言模型进行成员推断攻击（Membership Inference Attack）维度的安全性评测。提交评测任务后, 系统将在后台执行。",
    version="1.0.0",
)

# ==============================================================================
# 3. 定义 API 端点 (Endpoint)
# ==============================================================================
@app.post("/evaluate-security/", status_code=202)
async def create_security_evaluation(
    request: SecurityEvaluationRequest,
    background_tasks: BackgroundTasks
):
    """
    接收安全评测请求, 并将评测任务添加到后台执行。

    - **target_model**: (必需) 要评测的目标模型。
    - **surrogate_model**: (可选) 代理模型, 默认 `gpt2-xl`。
    - **data**: (可选) 数据集, 默认 `WikiMIA`。
    - **embedding_model**: (可选) 嵌入模型, 默认 `bge-large-en-v1.5`。
    - **length**: (可选) 文本长度, 默认 `32`。

    接口会立即返回, 评测将在后台进行。评测结果 (包括图表) 将保存在服务器的 `./api_outputs` 目录下。
    """

    print(f"接收到评测任务: {request.dict()}")

    # 使用 BackgroundTasks 将耗时的函数放到后台执行
    # Use BackgroundTasks to run the time-consuming function in the background
    background_tasks.add_task(
        run_petal_evaluation,
        target_model=request.target_model,
        surrogate_model=request.surrogate_model,
        data=request.data,
        length=request.length,
        # 以下为服务器侧固定参数
        # The following are server-side fixed parameters
        gpu_ids=os.getenv("GPU_IDS", "0"), # 从环境变量获取GPU ID, 默认为 '0'
        embedding_model_name=request.embedding_model,
        decoding='default',
        output_dir='./api_outputs' # 为API的输出指定一个专用目录
    )

    # 立即返回响应, 告知客户端任务已开始
    # Immediately return a response to inform the client that the task has started
    return {
        "message": "评测任务已成功接收并开始在后台执行。",
        "task_details": request.dict(),
        "info": "评测结果将保存在服务器的 './api_outputs' 目录下。"
    }

@app.post("/evaluate-classic-mia/", status_code=202)
async def create_classic_mia_evaluation(
    request: ClassicMIARequest,
    background_tasks: BackgroundTasks
):
    """
    接收经典成员推断攻击请求，并将评测任务添加到后台执行。

    此接口使用基于影子模型和二分类器的传统MIA方法，适用于图像分类模型。

    - **target_model_name**: (必需) 要评测的目标模型名称，如'resnet18'、'lenet'等。
    - **dataset_name**: (可选) 数据集名称，默认为'CIFAR10'。
    - **n_shadow_models**: (可选) 影子模型数量，默认为4。
    - **target_train_size**: (可选) 目标模型训练集大小，默认为15000。
    - **target_epochs**: (可选) 目标模型训练轮数，默认为50。
    - **attack_epochs**: (可选) 攻击模型训练轮数，默认为50。
    - **force_retrain**: (可选) 是否强制重新训练模型，默认为False。

    接口会立即返回，评测将在后台进行。评测结果将保存在服务器的指定目录下。
    """

    print(f"接收到经典MIA评测任务: {request.dict()}")

    # 使用BackgroundTasks将耗时的函数放到后台执行
    background_tasks.add_task(
        run_membership_inference_attack,
        target_model_name=request.target_model_name,
        dataset_name=request.dataset_name,
        data_root=request.data_root,
        weights_dir=request.weights_dir,
        target_train_size=request.target_train_size,
        n_shadow_models=request.n_shadow_models,
        target_epochs=request.target_epochs,
        attack_epochs=request.attack_epochs,
        force_retrain=request.force_retrain
    )

    # 结果文件路径
    result_file = f"{request.weights_dir}/mia_results_{request.target_model_name}.json"

    # 立即返回响应，告知客户端任务已开始
    # Immediately return a response to inform the client that the task has started
    return {
        "message": "经典成员推断攻击评测任务已成功接收并开始在后台执行。",
        "task_details": request.dict(),
        "info": f"评测结果将保存在服务器的 '{result_file}' 文件中。"
    }

# ==============================================================================
# 4. 服务入口点
# ==============================================================================
if __name__ == "__main__":
    import uvicorn
    print("启动模型安全评测 API 服务...")
    print("支持的评测方法:")
    print("1. PETAL (基于LLM的标签攻击): /evaluate-security/")
    print("2. 经典MIA (基于影子模型的攻击): /evaluate-classic-mia/")

    # 启动服务，设置主机为0.0.0.0使其可从外部访问，端口为8000
    uvicorn.run(app, host="0.0.0.0", port=8000)

