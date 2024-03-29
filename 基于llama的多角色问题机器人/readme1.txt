1--首先安装conda环境
	创建conda环境，进入环境
	使用命令conda create -n env_name --clone lab

2--启动  pip install -r requirement.txt  安装剩余环境

3--修改config_module.py来设置相关参数	
	llama文件为  Llama-2-7b-chat-hf
	下载到文件夹：https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

4--微调命令python train_lora.py  后面相关参数自己调整，具体在config_module.py里设置默认值，或者自己跑的时候后面加参数

如 python train_lora.py --base_model /root/LLM/Llama-2-7b-chat-hf \
    --tokenizer /root/LLM/Llama-2-7b-chat-hf \
    --dataset "data/text.json" \
    --dataset_format "alpaca" \


5--运行命令my_gradio.py  同样自己调整config_module.py

不出gradio的公用网页地址的话可以网上搜一下解决办法，下面的这篇博客或许有帮助
https://blog.csdn.net/qq_44193969/article/details/131966975?ops_request_misc=&request_id=&biz_id=102&utm_term=frpc_linux_amd64_v0.2&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-1-131966975.142^v100^pc_search_result_base4&spm=1018.2226.3001.4187

数据格式是json，preprocess文件可以生成对应格式的文本，SFT方法最好用
"text":"###Human:xxxxxxxxx###Assistant:xxxxxx"


可能会有新的warning信息：可以在config_module文件中class train_config(TrainingArguments)函数中加入
gradient_checkpointing_kwargs: List[bool] = field(default_factory=lambda: {"use_reentrant": True}, metadata={"help": 'debug for warning'})
