import json

origin_filename: str = "alpaca_data_cleaned_1000.json"
output_filename: str = "text.json"


with open(origin_filename, "r", encoding="utf-8") as fp:
    data = json.load(fp)

changed_data = []
for i in data:
    changed_data.append({"text": "### Human: " + i["instruction"] + "\n" + i["input"] + "\n### Assistant: " + i["output"]})


print(f" 已创建 {len(changed_data)} 个样例数据")
# 将数据保存为 JSON 文件

with open(output_filename, 'w', encoding='utf-8') as json_file:
    json.dump(changed_data, json_file, ensure_ascii=False, indent=4)


print(f" JSON 数据集已创建并保存为 {output_filename}。")
