import json

with open("imagenet_dataset_row.json", "r", encoding="utf-8") as f:
    orig_dict = json.load(f)
    print(orig_dict)


# 2) COCO 형식으로 변환
coco_list = []
for key, value in sorted(orig_dict.items(), key=lambda x: int(x[0])):
    idx = int(key) + 1   # ID는 1부터 시작
    name = value         # 혹은 value.split(",")[0].strip() 처럼 첫 번째 이름만 사용
    coco_list.append({"id": idx, "name": name})

coco_dict = {"COCO dataset": coco_list}

# 3) JSON으로 출력 (파일로 저장하거나 문자열로 얻기)
output_json_str = json.dumps(coco_dict, ensure_ascii=False, indent=2)
print(output_json_str)
with open("imagenet_dataset_like_coco.json", "w", encoding="utf-8") as f:
    json.dump(coco_dict, f, ensure_ascii=False, indent=2)
