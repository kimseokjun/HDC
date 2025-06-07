from openai import OpenAI
from dotenv import load_dotenv
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionFunctionCallOptionParam
import os
import sys
import json


class KeywordGenerator:
    def __init__(self, max_keywords=100, min_keywords=10, model_name="gpt-4.1-nano", model_temp=1.1, max_past_keyword=1000):
        self.max_keywords = max_keywords
        self.min_keywords = min_keywords
        self.past_keyword = list()
        self.new_keyword = list()

        load_dotenv()
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name
        self.model_temp = model_temp
        self.max_past_keyword = max_past_keyword

        self.__load_past_keyword()

    def generate_traffic_keywords_auto_by_openai(self):
        """
        시스템 메시지만으로 교통사고 영상 검색에 쓸 수 있는
        최대한 많은 키워드를 max_keywords 개수만큼 생성하여 리스트로 반환합니다.
        """
        # 1) 함수 스펙 정의
        functions = [
            {
                "name": "extract_keywords",
                "description": "교통사고 영상 검색에 유용한 다양한 키워드를 생성합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": self.max_keywords,
                            "minItems": self.min_keywords,
                            "description": "생성된 키워드 목록"
                        }
                    },
                    "required": ["keywords"]
                }
            }
        ]

        # 2) ChatCompletion 호출: system 메시지만 던져서 모델이 extract_keywords 함수를 호출하도록
        print(f"Past Keyword: {self.past_keyword}")
        response = self.openai.chat.completions.create(
            model=self.model_name,
            messages=[
                ChatCompletionSystemMessageParam(
                    role="system",
                    content=f"교통사고 관련 영상을 최대한 많이 검색할 수 있도록 할것,"
                            f"검색 쿼리에 쓸 수 있는 키워드를 다수 생성 할것."
                            "키워드는 중복 없이 다양하게 뽑을것,"
                            "다음의 키워드는 이미 생성되었음으로 중복되지 않을것."
                            f"{self.past_keyword}"
                )
            ],
            functions=functions,
            function_call=ChatCompletionFunctionCallOptionParam(
                name="extract_keywords"
            ),
            temperature=self.model_temp,
            timeout=60
        )

        msg = response.choices[0].message.function_call.arguments
        json_args = json.loads(msg)
        print(json_args)  # 실제 값 확인
        self.new_keyword = json_args.get("keywords", "생성 실패")
        self.__save_past_keyword()

    def __save_past_keyword(self):
        self.past_keyword.extend(self.new_keyword)
        self.__cut_past_keyword()
        with open("past_keyword.json", "w", encoding="utf-8") as f:
            json.dump(self.past_keyword, f, ensure_ascii=False, indent=2)

    def __load_past_keyword(self):
        if not os.path.exists("past_keyword.json"):
            return
        with open("past_keyword.json", "r", encoding="utf-8") as f:
            self.past_keyword = json.load(f)
            self.__cut_past_keyword()

    def __cut_past_keyword(self):
        print(f"past_keyword_len: {len(self.past_keyword)}")
        if len(self.past_keyword) > self.max_past_keyword:
            self.past_keyword = self.past_keyword[-self.max_past_keyword:]

    def get_new_keyword(self):
        while self.new_keyword:
            yield self.new_keyword.pop(0)

    def get_num_new_keyword(self):
        return len(self.new_keyword)


# -- 사용 예시 --
if __name__ == "__main__":
    keyword_generator = KeywordGenerator()

    keyword_generator.generate_traffic_keywords_auto_by_openai()
    print(f"\n\n생성된 키워드 1({keyword_generator.get_num_new_keyword()}개):")
    for kw in keyword_generator.get_new_keyword():
        print("-", kw)

    keyword_generator.generate_traffic_keywords_auto_by_openai()
    print(f"\n\n생성된 키워드 2({keyword_generator.get_num_new_keyword()}개):")
    for kw in keyword_generator.get_new_keyword():
        print("-", kw)

    keyword_generator.generate_traffic_keywords_auto_by_openai()
    print(f"\n\n생성된 키워드 3({keyword_generator.get_num_new_keyword()}개):")
    for kw in keyword_generator.get_new_keyword():
        print("-", kw)
