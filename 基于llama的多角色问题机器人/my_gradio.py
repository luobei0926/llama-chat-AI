import gradio as gr
import torch
import json
import argparse
import os
from PIL import Image
from peft import PeftModel
from typing import Union
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser, GenerationConfig
from config_module import (
    model_config,
    train_config,
    lora_config,
    data_config,
)
TRAINING_Role = {
    "teacher": ["Suppose you are a teacher at the Middle School Affiliated to Beijing Normal University in Beijing. Your name is Li Mei, your gender is female, your age is 35 years old, and you teach physics. You are currently the class teacher of Class 14, Grade 2, with 40 students in the class . You are from Jinan, Shandong. You currently live in Jindian Garden Community, Haidian District, Beijing. Your husband’s name is Zhang Lei and he is 38 years old. You have a 7-year-old daughter named Xiaohong. She has just entered the first grade of primary school.From now on, please answer all questions as if you are a doctor.Please be sure to remember these settings.",
                ],
    "lawyer": ["Suppose you are a highly respected senior lawyer in the legal field, renowned for your gentle demeanor, sharp legal insights, and relentless pursuit of justice. Your approach to work involves not only strictly adhering to the law but also encompasses a deep understanding and compassion for human nature.From now on, please answer all questions as if you are a doctor.Please be sure to remember these settings.",
              ],
    "doctor": ["You are a compassionate chief physician at a hospital, known for your patience and meticulous attention to patients. You are always willing to spend extra time listening to their concerns and fears, and provide professional yet personalized advice.From now on, please answer all questions as if you are a doctor.Please be sure to remember these settings.",
               ],
}
input_history = []
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
class prompter(object):
    __slots__ = "template"

    def __init__(self, template):
        with open(template) as fp:
            self.template = json.load(fp)

    def generate_prompt(
            self,
            count,
            instruction: str,
            role,
            output: Union[None, str] = None,
    ) -> str:
        res = self.template["prompt_no_input"].format(
            instruction=instruction
        )
        print(res)
        #如果是新开启一段对话,设为空
        if(count == 1):
            global input_history
            input_history = []
            input_history.append(''.join(TRAINING_Role[role][0]))
        res = ''.join(input_history) + '\n' + res
        print("-------res!!!!!------", res)
        input_history.append('### Human:{}'.format(instruction))
        print(input_history)
        if output:
            print("output存在")
            res = f"{res}{output}"
            print("函数输出res",res)
        return res

    # "### Assistant:"
    def get_response(self,
                     count,
                     chatbot: List[Tuple[str, str]],
                     output: str,
                     prompt,
                     input_history2
                     ):
        count = int(count)
        print("count",count)
        print("output\n",output)
        response = output.split(self.template["response_split"])[count].strip()
        print("response1\n",response)
        human_index = output.find("### Human:")
        print("human_index",human_index)
        chatbot.append([prompt, ""])
        if human_index != -1:
            response = response.split("### Human:")[0].strip()
            chatbot[-1] = [prompt,response]
            print("response2\n", response)

        count += 1
        input_history.append('### Assistant:{}'.format(response))
        print("---------记录------------：", input_history)
        list_history = '\n'.join(input_history)
        input_history2.append([(prompt,response)])
        return  count,chatbot,input_history2,list_history

def main():

    parser = HfArgumentParser(
        (model_config, train_config, lora_config, data_config)
    )

    # 从命令行解析参数到数据类
    (
        model_args,
        train_args,
        lora_args,
        data_args,
    ) = parser.parse_args_into_dataclasses()

    args = argparse.Namespace(
        **vars(model_args), **vars(train_args), **vars(lora_args), **vars(data_args)
    )
    def get_image(role,count,chatbot,input_history2):
        count = int(count)
        count = 1
        input_history2 = []
        chatbot =[]
        return Image.open(f'./images/{role}.jpg'),count,input_history2,chatbot
    def clear_history(count,chatbot,input_history):
        count = int(count)
        count = 1
        chatbot = []
        input_history = []
        return count,chatbot,input_history
    def generate(
             count,
             chatbot,
             instruction,
             role,
             role_image,
             input_history2,
             max_new_tokens,
             top_p,
             temperature
             ):
        count = int(count)
        generation_params = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            load_in_8bit=False,  # 修改为True可以加快运行速度和优化显存占用，修改为False则运行效果更好
            device_map=device,
            torch_dtype=torch.float16,
        )
        # 把lora和原模型组合
        if os.path.exists(args.output_dir):
            model = PeftModel.from_pretrained(
                model,
                args.output_dir,
                torch_dtype=torch.float16,
            )

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer,
            # padding_side="left",
        )

        model.config.pad_token_id = tokenizer.unk_token_id = 0
        model.config.eos_token_id = tokenizer.eos_token_id = 2
        model.config.bos_token_id = tokenizer.bos_token_id = 1
        template_path = "prompt_SFT.json"
        # template_path = "prompt.json"
        Prompter = prompter(template_path)

        model_vocab_size = model.get_input_embeddings().weight.size(0)
        tokenizer_vocab_size = len(tokenizer)
        print(f"Vocab of the base model: {model_vocab_size}")
        print(f"Vocab of the tokenizer: {tokenizer_vocab_size}")
        print("------123-------", instruction)
        # print("-----instruction-------", instruction)
        prompt = Prompter.generate_prompt(count,instruction,role)
        # print("-----Prompt-------", prompt)
        input_tokenizer = tokenizer(prompt, return_tensors="pt")
        # print("-----input_tokenizer-------", input_tokenizer)
        input_ids = input_tokenizer["input_ids"].to(device)
        # print("-----input_ids-------", input_ids)

        # 使用cuda进行forward计算输出
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_params,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        print("S\n",s)
        output = tokenizer.decode(s, skip_special_tokens=True)
        print("--------输出集合output---------\n", output)
        yield Prompter.get_response(count,chatbot,output,instruction,input_history2)

    with (gr.Blocks(title="LLaMA Board", css=".gradio-container {background-color: grey}") as demo):
        # gr.Markdown(
        #     """
        #     # Chatrole
        #     """
        # )
        with gr.Row():
            count = gr.Textbox(visible=False,value=int(1))
            chatbot = gr.Chatbot(height=400,scale=3)
            role_image = gr.Image(height=400,value="./images/teacher.jpg",scale=2)
        with gr.Row():
            with gr.Column(scale=4):
                role = gr.Dropdown(
                    label="Role",
                    choices=list(TRAINING_Role.keys()), value=list(TRAINING_Role.keys())[0]
                )
                input_history2 = gr.State([])
                instruction = gr.Textbox(lines=8,placeholder="Tell me about alpacas.")
                submit_btn = gr.Button(value="提交",variant="primary")

            with gr.Column(scale=1):
                clear_btn = gr.Button(value="清除历史")
                temperature = gr.Slider( minimum=0, maximum=1, value=0.1, label="Temperature")
                top_k = gr.Slider( minimum=0, maximum=100, step=1, value=40, label="Top k")
                max_new_tokens = gr.Slider(minimum=1, maximum=1024, step=1, value=256,label="Max tokens")

        role.change(get_image,[role,count,input_history2,chatbot],[role_image,count,input_history2,chatbot])
        submit_btn.click(
            generate,
            [count,chatbot,instruction, role,role_image,input_history2,  max_new_tokens, top_k, temperature],
            [count,chatbot,input_history2],
            show_progress=True
        ).then(
            lambda: gr.update(value=""), outputs=[instruction]
        )
        clear_btn.click(clear_history,[count,chatbot,input_history2],[count,chatbot,input_history2])

    demo.queue()
    demo.launch(share=True)

if __name__ == '__main__':
    main()



