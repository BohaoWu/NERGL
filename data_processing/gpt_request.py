from collections import Counter
from openai import OpenAI
import anthropic

import os
import time

# ne_info_str gpt_ne_info_with_example
# ner_result_str gpt_ner_res_4_1
# ner_tag_result gpt_ner_res_4_1_tag
# ner_tag_result_without_hint gpt_ner_res_4_1_tag_without_hint


class GPTClient:
    
    def __init__(self, gpt_model=None):
        # gpt
        self.gpt_model = gpt_model or os.environ.get("RAG_GPT_MODEL", "gpt-4.1")
        self.snapshots = Counter()
        self.gpt_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        # claude
        self.claude_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        
        
        self.prompt = "あなたは20年以上の経験を持つ浮世絵の専門家です。\n"

        # The four types, each with one line of explanation. 替名 is a role name
        # inside a play and is easily confused with the stage name of the actor
        # who performs it, so the two are contrasted in every example below.
        self.type_guide = "固有表現の種類は次の四つです。\n\
                            ・地名：日本の土地や場所の名称。\n\
                            ・演目：歌舞伎や浄瑠璃などの作品の題名。\n\
                            ・替名：作品の中の登場人物の役名。演じる役者の芸名とは別のものです。\n\
                            ・役者：歌舞伎役者の芸名。\n"

        # Every example entity below appears only in the training split, so none
        # of them discloses an answer from the development or the test split.
        self.ne_info_str = "システム:あなたは20年以上の経験を持つ浮世絵の専門家です。以下の例にならって答えてください。\n\
                            浮世絵タイトル：「名所隅田川八景」 「吉原仲之町」\n\
                            質問：「上記の浮世絵タイトルにおいて、「隅田川」 「吉原」とは何ですか？修正を加えずに、簡潔に一文で答えてください。」\n\
                            答え：「隅田川 江戸の東を流れる川の名です。」 「吉原 江戸の遊廓があった場所の名です。」\n\
                            浮世絵タイトル：「忠臣蔵」 「おかる 坂東家橘」\n\
                            質問：「上記の浮世絵タイトルにおいて、「忠臣蔵」 「おかる」 「坂東家橘」とは何ですか？修正を加えずに、簡潔に一文で答えてください。」\n\
                            答え：「忠臣蔵 歌舞伎の演目の題名です。」 「おかる 忠臣蔵に登場する人物の役名です。」 「坂東家橘 歌舞伎役者の芸名です。」\n\
                            浮世絵タイトル：「関兵衛 市川海老蔵」\n\
                            質問：「上記の浮世絵タイトルにおいて、「関兵衛」 「市川海老蔵」とは何ですか？修正を加えずに、簡潔に一文で答えてください。」\n\
                            答え：「関兵衛 作品に登場する人物の役名です。」 「市川海老蔵 歌舞伎役者の芸名です。」\n"

        self.ner_result_str = "システム:あなたは20年以上の経験を持つ浮世絵の専門家です。以下の例にならって答えてください。\n\
                            浮世絵タイトル：「名所隅田川八景」 「吉原仲之町」\n\
                            質問：「上記の浮世絵タイトルにおいて、「隅田川」 「吉原」とは何ですか？修正を加えずに、簡潔に一文で答えてください。」\n\
                            答え：「隅田川 地名」 「吉原 地名」\n\
                            浮世絵タイトル：「忠臣蔵」 「おかる 坂東家橘」\n\
                            質問：「上記の浮世絵タイトルにおいて、「忠臣蔵」 「おかる」 「坂東家橘」とは何ですか？修正を加えずに、簡潔に一文で答えてください。」\n\
                            答え：「忠臣蔵 演目」 「おかる 替名」 「坂東家橘 役者」\n\
                            浮世絵タイトル：「関兵衛 市川海老蔵」\n\
                            質問：「上記の浮世絵タイトルにおいて、「関兵衛」 「市川海老蔵」とは何ですか？修正を加えずに、簡潔に一文で答えてください。」\n\
                            答え：「関兵衛 替名」 「市川海老蔵 役者」\n"

        self.ner_result_tag_str = "システム:あなたは20年以上の経験を持つ浮世絵の専門家です。以下の例にならって答えてください。\n\
                            " + self.type_guide + "\
                            浮世絵タイトル：「名所隅田川八景」 「吉原仲之町」\n\
                            質問：「上記の浮世絵タイトルにおいて、「隅田川」 「吉原」とは地名、演目、替名、役者のどちらですか？修正を加えずに、簡潔に一行で答えてください。」\n\
                            答え：「地名 地名」\n\
                            浮世絵タイトル：「忠臣蔵」 「おかる 坂東家橘」\n\
                            質問：「上記の浮世絵タイトルにおいて、「忠臣蔵」 「おかる」 「坂東家橘」とは地名、演目、替名、役者のどちらですか？修正を加えずに、簡潔に一行で答えてください。」\n\
                            答え：「演目 替名 役者」\n\
                            浮世絵タイトル：「関兵衛 市川海老蔵」\n\
                            質問：「上記の浮世絵タイトルにおいて、「関兵衛」 「市川海老蔵」とは地名、演目、替名、役者のどちらですか？修正を加えずに、簡潔に一行で答えてください。」\n\
                            答え：「替名 役者」\n"

        self.ner_result_tag_str_with_hint = "システム:あなたは20年以上の経験を持つ浮世絵の専門家です。以下の例にならって答えてください。\n\
                            " + self.type_guide + "\
                            浮世絵タイトル：「名所隅田川八景」 「吉原仲之町」\n\
                            質問：「上記の浮世絵タイトルにおいて、地名、演目、替名、役者は何ですか？修正を加えずに、簡潔に一文で答えてください。」\n\
                            答え：「隅田川 地名」 「吉原 地名」\n\
                            浮世絵タイトル：「忠臣蔵」 「おかる 坂東家橘」\n\
                            質問：「上記の浮世絵タイトルにおいて、地名、演目、替名、役者は何ですか？修正を加えずに、簡潔に一文で答えてください。」\n\
                            答え：「忠臣蔵 演目」 「おかる 替名」 「坂東家橘 役者」\n\
                            浮世絵タイトル：「関兵衛 市川海老蔵」\n\
                            質問：「上記の浮世絵タイトルにおいて、地名、演目、替名、役者は何ですか？修正を加えずに、簡潔に一文で答えてください。」\n\
                            答え：「関兵衛 替名」 「市川海老蔵 役者」\n"

        return

    def getNERResultStrResponseFomGPT(self, sentence):
        pattern = f"{self.prompt} \
                {self.ner_result_tag_str_with_hint} \
                浮世絵タイトル: '{sentence}'\n \
                質問：「上記の浮世絵タイトルにおいて、地名、演目、替名、役者は何ですか？修正を加えずに、簡潔に一文で答えてください。」\n"
        
        response = self.getResponseFromGPT(pattern=pattern)
        return response

    def getResponseFromGPT(self, pattern):
        try:
            response = self.gpt_client.chat.completions.create(model=self.gpt_model,messages=[{"role":"user","content": pattern},])
        except:
            time.sleep(0.1)
            response = self.gpt_client.chat.completions.create(model=self.gpt_model,messages=[{"role":"user","content": pattern},])

        # The requested name is an alias; record the snapshot that served it.
        self.snapshots[response.model] += 1
        response = response.choices[0].message.content
        return response
    
    
    ########################### Claude ##############################
    
    def getNERResultStrResponseFomClaude(self, pattern):
        pattern = f"{self.prompt} \
                {self.ner_result_tag_str_with_hint} \
                浮世絵タイトル: '{pattern}'\n \
                質問：「上記の浮世絵タイトルにおいて、地名、演目、替名、役者は何ですか？修正を加えずに、簡潔に一文で答えてください。」\n"
        
        response = self.getResponseFromClaude(pattern=pattern)
        return response
    
    def getResponseFromClaude(self, pattern):
        message = ""
        try:
            message = self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": pattern
                    }
                ]
            )
            # 检查是否有内容
            if message.content and len(message.content) <= 0:
                print("没有收到响应内容")
   
        except Exception as e:
            print(f"发生错误: {e}")
        return message.content[0].text
    
    
    
    def getDictfromFile(self, file):
        rag_dict = {}
        with open(file, "r") as f:
            for line in f.readlines:
                splited_line = line.splited("###")
                id = splited_line[0]
                rag_words = splited_line[1]
                rag_dict[id] = rag_words
        return rag_dict
        

if __name__ == "__main__":

    pattern = "「名所江戸百景」 「目黒爺々が茶屋」"
    entity = ["江戸"]
    prompt = ""
            
    gpt_client = GPTClient()
    pattern = "「安達元右衛門　尾上菊五郎」「椀助　尾上梅五郎」"
    entity = ["安達元右衛門", "尾上菊五郎", "椀助", "尾上梅五郎"]

    response = gpt_client.getNERResultStrResponseFomClaude(pattern=pattern)
    print(response)

