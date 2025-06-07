from openai import OpenAI
from dotenv import load_dotenv
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionFunctionCallOptionParam
import os
import sys
import json


class IsContainDataset:
    def __init__(self, keyword, model_name="gpt-4.1-mini", model_temp=0.9, max_request_count=30):
        self.keyword = keyword
        load_dotenv()
        self.dataset_classes_json = dict()
        self.load_dataset_classes_list("dataset_classes.json")
        self.max_request_count = max_request_count
        self.request_count = 0

        self.dataset_names = list(self.dataset_classes_json.keys())
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name
        self.model_temp = model_temp

        self.message_history = list()
        self.keyword_history = list()

    def load_dataset_classes_list(self, dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            self.dataset_classes_json = json.load(f)

    def __get_message_history(self, with_system_prompt=True):
        message_history_with_prompt = self.message_history.copy()
        if with_system_prompt:
            message_history_with_prompt.append(ChatCompletionSystemMessageParam(
                role="system",
                content=f"머신러닝에 사용하기 위한 데이터셋을 조회할 것\n"+
                        f"사용자가 입력한 키워드는 다음과 같다: {self.keyword}\n"+
                        f"위 키워드를 DataSet에 펑션콜을 통해 조회하여 나온 결과를 바탕으로 결정할것\n"+
                        f"모든 데이터셋은 영문자로만 이루어져 있음"+
                        f"적절한 펑션콜로만 응답할것\n"
        ))
        return message_history_with_prompt

    def __get_function_call_param(self):
        functions = [
            {
                "name": "extract_dataset_classes",
                "description": "사용자가 요구하는 키워드가 데이터셋에 포함되어 있는지 판단",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "is_contain": {
                            "type": "boolean",
                            "description": "데이터셋에 포함되어 있으면 true, 없으면 false"
                        },
                        "what_dataset": {
                            "type": "string",
                            "description": "포함되어있는 DataSet 명, 없으면 None"
                        },
                        "what_id": {
                            "type": "integer",
                            "description": "해당되는 DataSet Class의 ID, 없으면 -1"
                        }
                    },
                    "required": ["is_contain", "what_dataset", "what_id"]
                }
            },
            {
                "name": "search_dataset_classes",
                "description": "Dataset과 이름, id를 검색",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_keyword": {
                            "type": "string",
                            "description": f"{self.dataset_names}에서 키워드를 검색하여 리턴"
                        }
                    },
                    "required": ["search_keyword"]
                }
            },

        ]
        return functions

    def request_dataset_by_openai(self):
        """
        시스템 메시지만으로 교통사고 영상 검색에 쓸 수 있는
        최대한 많은 키워드를 max_keywords 개수만큼 생성하여 리스트로 반환합니다.
        """
        # 1) 함수 스펙 정의


        print("\nrequest_dataset_by_openai:")
        print(self.__get_message_history())
        self.request_count += 1
        if self.request_count >= self.max_request_count:
            raise Exception("최대 요청 횟수를 초과했습니다.")
        response = self.openai.chat.completions.create(
            model=self.model_name,
            messages=self.__get_message_history(),
            functions=self.__get_function_call_param(),
            function_call="auto",
            temperature=self.model_temp,
            timeout=60
        )


        msg = response.choices[0].message.function_call.arguments
        print("Row_return" + msg)  # 실제 값 확인
        self.message_history.append(response.choices[0].message)
        json_args = json.loads(msg)


        if json_args.get("is_contain", None) is not None:
            print(f"추출 결과: {json_args}")

            if not self.handling_response_is_valid_extract_dataset_classes(
                json_args.get("is_contain", None), json_args.get("what_dataset", None), json_args.get("what_id", None)):
                return self.request_dataset_by_openai()
            else:
                return_dict={"is_contain": json_args.get("is_contain", None),
                        "what_dataset": json_args.get("what_dataset", None),
                        "what_id": json_args.get("what_id", None),
                        "name": self.get_name_from_id(json_args.get("what_dataset", None), json_args.get("what_id", None))
                        }
                return return_dict

        elif json_args.get("search_keyword", None) is not None:
            search_keyword = json_args.get("search_keyword")
            try:
                search_results = self.handling_response_search_dataset_classes(search_keyword)
                self.message_history.append(ChatCompletionSystemMessageParam(
                    role="system",
                    content=f"검색 키워드: {search_keyword}\n"+
                            f"검색 결과: \n{search_results}"
                ))
                print(f"검색 결과: \n{search_results}")
                return self.request_dataset_by_openai()
            except ValueError as e:
                print(f"OpenAI API 검색 중 예외 발생: {e}")
                return f"Error: {e}"
        else:
            return None


    def __is_not_english(self, text):
        """
        주어진 텍스트에 영어가 아닌 문자가 포함되어 있는지 확인하는 메서드

        Args:
            text (str): 확인할 텍스트

        Returns:
            bool: 영어가 아닌 문자가 포함되어 있으면 True, 아니면 False
        """
        # 영어 문자 범위: a-z, A-Z, 0-9, 공백 및 일반적인 구두점

        for char in text:
            # 영어 알파벳, 숫자, 공백, 일반적인 구두점이 아닌 경우
            if not (('a' <= char <= 'z') or ('A' <= char <= 'Z') or 
                    ('0' <= char <= '9') or char in ' .,!?-_\'":;()[]{}'):
                return True
        return False

    def handling_response_search_dataset_classes(self, search_keyword):
        """
        dataset_classes_json에서 name을 검색하는 메서드
        영어가 아닌 키워드가 입력되면 예외를 발생시킵니다.

        Args:
            search_keyword (str): 검색할 키워드

        Returns:
            list: 검색 결과 리스트 (dataset_name, class_id, class_name)

        Raises:
            ValueError: 키워드가 영어가 아닌 경우 발생
        """
        results = []

        # 영어가 아닌 키워드 체크
        if self.__is_not_english(search_keyword):
            print("영어가 아닌 키워드는 지원하지 않음.")
            self.message_history.append(ChatCompletionSystemMessageParam(
                role="system",
                content=f"영어가 아닌 키워드는 지원하지 않음. 검색 키워드: {search_keyword}. 영문자로 검색할 것"
            ))
            return results

        if search_keyword in self.keyword_history:
            print("이미 검색한 키워드입니다.")
            self.message_history.append(ChatCompletionSystemMessageParam(
                role="system",
                content=f"이미 검색한 키워드. 검색 키워드: {search_keyword}. 다른 키워드로 검색할 것"
            ))
            return results


        self.keyword_history.append(search_keyword)

        for dataset_name, classes in self.dataset_classes_json.items():
            for class_obj in classes:
                if search_keyword.lower() in class_obj["name"].lower():
                    results.append({
                        "dataset": dataset_name,
                        "id": class_obj["id"],
                        "name": class_obj["name"]
                    })

        if not results:
            self.message_history.append(ChatCompletionSystemMessageParam(
                role="system",
                content=f"검색 키워드({search_keyword})의 검색 결과가 없음. 다른 검색어를 사용할 것"
            ))

        return results

    def handling_response_is_valid_extract_dataset_classes(self, is_contain, what_dataset, what_id):
        self.message_history.append(ChatCompletionSystemMessageParam(
            role="system",
            content=f"추출 결과: is_contain: {is_contain}, what_dataset: {what_dataset}, what_id: {what_id}"
        ))
        try:
            if is_contain is False:
                return True

            if what_dataset is None or what_dataset == "" or what_id == -1:
                raise ValueError("what_dataset is None or what_dataset == "" or what_id == -1")
            elif what_dataset not in self.dataset_classes_json:
                raise ValueError(f"what_dataset not in self.dataset_classes_json: {what_dataset}")
            elif what_id < 0 or what_id >= len(self.dataset_classes_json[what_dataset]):
                raise ValueError(f"what_id < 0 or what_id >= len(self.dataset_classes_json[what_dataset]): {what_id}")
            return True


        except ValueError as e:
            print(f"추출 결과가 잘못되었습니다. {e}")
            self.message_history.append(ChatCompletionSystemMessageParam(
                role="system",
                content=f"추출 결과: 잘못 추출된 데이터셋 -> e"
            ))
            return False

    def get_name_from_id(self, dataset_name, class_id):
        try:
            for class_obj in self.dataset_classes_json[dataset_name]:
                if class_obj["id"] == class_id:
                    return class_obj["name"]
            return None
        except ValueError as e:
            print(f"get_name_from_id ValueError: {e}")
            return None
        except Exception as e:
            print(f"get_name_from_id: {e}")
            return None





# -- 사용 예시 --
if __name__ == "__main__":
    # 테스트 1: OpenAI API를 통한 검색
    print("=== OpenAI API를 통한 검색 ===")
    while True:
        try:
            input_keyword = input("검색할 키워드를 입력하세요: ")
            is_contain = IsContainDataset(keyword=input_keyword)
            result = is_contain.request_dataset_by_openai()
            print("**결과**")
            print(result)
        except KeyboardInterrupt:
            print("Program terminated by user.")
            sys.exit(0)
        except Exception as e:
            print(f"OpenAI API 검색 태스트 중 예외 발생: {e}")

