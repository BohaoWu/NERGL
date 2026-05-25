import re
from openai import OpenAI
import anthropic

import os
import time

# ne_info_str gpt_ne_info_with_example
# ner_result_str gpt_ner_res_4_1
# ner_tag_result gpt_ner_res_4_1_tag
# ner_tag_result_without_hint gpt_ner_res_4_1_tag_without_hint


class GPTClient:
    
    def __init__(self):
        # gpt
        self.gpt_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        # claude
        self.claude_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        
        
        self.prompt = "あなたは20年以上の経験を持つ浮世絵と情報学の専門家です。\n"      
        self.ne_info_str = "システム:あなたは20年以上の経験を持つ浮世絵と情報学の専門家です。以下の例にならって回答してください。\n\
                            浮世絵タイトル：「名所江戸百景」 「目黒爺々が茶屋」\n\
                            質問：「上記の浮世絵タイトルにおいて、「江戸」 「目黒」とは何ですか？修正を加えずに、簡潔に一文を返事してください。」\n\
                            答え：「江戸 現在の東京都にあたる、江戸時代の日本の都市名。 「目黒 江戸時代の江戸郊外に位置する地名です」 \n\
                            浮世絵タイトル：「仁木弾正 尾上菊五郎」\n\
                            質問：「上記の浮世絵タイトルにおいて、「仁木弾正」とは何ですか？修正を加えずに、簡潔に一文だけを返事してください。」\n\
                            答え：「仁木弾正 歌舞伎の演目に登場する架空の悪役武将。」 「尾上菊五郎 江戸時代から続く名跡を持つ歌舞伎役者の名前です。」\n\
                            浮世絵タイトル：「吉岡帯刀」「市川団十郎」\n\
                            質問：「上記の浮世絵タイトルにおいて、「吉岡帯刀」とは何ですか？修正を加えずに、簡潔に一文だけを返事してください。」\n\
                            答え：「吉岡帯刀 歌舞伎の演目に登場する架空の登場人物です。」 「市川団十郎 江戸時代から続く名跡を持つ歌舞伎役者の名です。」\n"
        
        self.ner_result_str = "システム:あなたは20年以上の経験を持つ浮世絵と情報学の専門家です。以下の例にならって回答してください。\n\
                            浮世絵タイトル：「名所江戸百景」 「目黒爺々が茶屋」\n\
                            質問：「上記の浮世絵タイトルにおいて、「江戸」 「目黒」とは何ですか？修正を加えずに、簡潔に一文を返事してください。」\n\
                            答え：「江戸 地名」 「目黒 地名」\n\
                            浮世絵タイトル：「仁木弾正 尾上菊五郎」\n\
                            質問：「上記の浮世絵タイトルにおいて、「仁木弾正」 「尾上菊五郎」とは何ですか？修正を加えずに、簡潔に一文だけを返事してください。」\n\
                            答え：「仁木弾正 替名」 「尾上菊五郎 役者」\n\
                            浮世絵タイトル：「吉岡帯刀」「市川団十郎」\n\
                            質問：「上記の浮世絵タイトルにおいて、「吉岡帯刀」 「市川団十郎」とは何ですか？修正を加えずに、簡潔に一文だけを返事してください。」\n\
                            答え：「吉岡帯刀 替名」 「市川団十郎 役者」\n\
                            テキスト：「「花合春之取組」 「小柳 尾上菊五郎」」\n\
                            質問：「上記の浮世絵タイトルにおいて、「花合春之取組」 「小柳」 「尾上菊五郎」とは何ですか？修正を加えずに、簡潔に一文だけを返事してください。」\n\
                            答え：「花合春之取組 演目」「小柳 替名」「尾上菊五郎 役者名」\n"
                            
        self.ner_result_tag_str = "システム:あなたは20年以上の経験を持つ浮世絵と情報学の専門家です。以下の例にならって回答してください。\n\
                            浮世絵タイトル：「名所江戸百景」 「目黒爺々が茶屋」\n\
                            質問：「上記の浮世絵タイトルにおいて、「江戸」 「目黒」とは地名、演目、替名、役者の中でどちですか？修正を加えずに、簡潔に一文を返事してください。」\n\
                            答え：「地名 地名」\n\
                            浮世絵タイトル：「仁木弾正 尾上菊五郎」\n\
                            質問：「上記の浮世絵タイトルにおいて、「仁木弾正」 「尾上菊五郎」とは地名、演目、替名、役者の中でどちですか？修正を加えずに、簡潔に一文だけを返事してください。」\n\
                            答え：「替名 役者」\n\
                            浮世絵タイトル：「吉岡帯刀」「市川団十郎」\n\
                            質問：「上記の浮世絵タイトルにおいて、「吉岡帯刀」 「市川団十郎」とは地名、演目、替名、役者の中でどちですか？修正を加えずに、簡潔に一文だけを返事してください。」\n\
                            答え：「替名 役者」\n\
                            テキスト：「「花合春之取組」 「小柳 尾上菊五郎」」\n\
                            質問：「上記の浮世絵タイトルにおいて、「花合春之取組」 「小柳」 「尾上菊五郎」とは地名、演目、替名、役者の中でどちですか？修正を加えずに、簡潔に一文だけを返事してください。」\n\
                            答え：「演目 替名 役者名」\n"    
                            
        self.ner_result_tag_str_with_hint = "システム:あなたは20年以上の経験を持つ浮世絵と情報学の専門家です。以下の例にならって回答してください。\n\
                            浮世絵タイトル：「名所江戸百景」 「目黒爺々が茶屋」\n\
                            質問：「上記の浮世絵タイトルにおいて、地名、演目、替名、役者は何ですか？修正を加えずに、簡潔に一文を返事してください。」\n\
                            答え：「江戸 地名」 「目黒 地名」\n\
                            浮世絵タイトル：「仁木弾正 尾上菊五郎」\n\
                            質問：「上記の浮世絵タイトルにおいて、地名、演目、替名、役者は何ですか？修正を加えずに、簡潔に一文だけを返事してください。」\n\
                            答え：「仁木弾正 替名」 「尾上菊五郎 役者」\n\
                            浮世絵タイトル：「吉岡帯刀」「市川団十郎」\n\
                            質問：「上記の浮世絵タイトルにおいて、地名、演目、替名、役者は何ですか？修正を加えずに、簡潔に一文だけを返事してください。」\n\
                            答え：「吉岡帯刀 替名」 「市川団十郎 役者」\n\
                            テキスト：「「花合春之取組」 「小柳 尾上菊五郎」」\n\
                            質問：「上記の浮世絵タイトルにおいて、地名、演目、替名、役者は何ですか？修正を加えずに、簡潔に一文だけを返事してください。」\n\
                            答え：「花合春之取組 演目」「小柳 替名」「尾上菊五郎 役者名」\n"
                            
        return

    def getNERExplainResponseFomGPT(self, sentence, entity):
        # responses = []
        entity_str = ""
        for index, i in enumerate(entity):
            if index != 0:
                entity_str + " "
            entity_str += "「" + i + "」"
        pattern = f"{self.prompt} \
                    {self.ukiyoe_str} \
                    浮世絵タイトル: '{sentence}'\n \
                    質問：「上記の浮世絵タイトルにおいて、{entity_str}は何ですか？修正を加えずに、簡潔に一列だけでを返事してください。」"
        response = self.getResponseFromGPT(pattern=pattern)
        # responses.append(response.choices[0].message.content)
        return response
    
    def getNERResultResponseFomGPT(self, sentence, entity):
        # responses = []
        entity_str = ""
        for index, i in enumerate(entity):
            if index != 0:
                entity_str + " "
            entity_str += "「" + i + "」"
        pattern = f"{self.prompt} \
                    {self.ner_result_str} \
                    {self.ukiyoe_str} \
                    浮世絵タイトル: '{sentence}'\n \
                    質問：「上記の浮世絵タイトルにおいて、{entity_str}は何ですか？修正を加えずに、簡潔に一列だけでを返事してください。」"
        response = self.getResponseFromGPT(pattern=pattern)
        # responses.append(response.choices[0].message.content)
        return response
    
    def getNERTagResultResponseFomGPT(self, sentence, entity):
        # responses = []
        entity_str = ""
        for index, i in enumerate(entity):
            if index != 0:
                entity_str + " "
            entity_str += "「" + i + "」"
        pattern = f"{self.prompt} \
                    {self.ner_result_str} \
                    浮世絵タイトル: '{sentence}'\n \
                    質問：「上記の浮世絵タイトルにおいて、{entity_str}とは地名、演目、替名、役者の中でどちですか？修正を加えずに、簡潔に一列だけでを返事してください。」"
        response = self.getResponseFromGPT(pattern=pattern)
        # responses.append(response.choices[0].message.content)
        return response

    def getNERResultStrResponseFomGPT(self, sentence):
        pattern = f"{self.prompt} \
                {self.ner_result_tag_str_with_hint} \
                浮世絵タイトル: '{sentence}'\n \
                質問：「上記の浮世絵タイトルにおいて、地名、演目、替名、役者は何ですか？修正を加えずに、簡潔に一文だけを返事してください。」\n"
        
        response = self.getResponseFromGPT(pattern=pattern)
        return response

    def getNERResponseFomGPT(self, sentence):
        pattern = f"{self.prompt} \
                浮世絵タイトル: '{sentence}'\n \
                質問：このタイトルにある役者、役目、地名に関することを、修正を加えずに、簡潔に一行だけで説明してください。」\n"
        
        response = self.getResponseFromGPT(pattern=pattern)
        return response
    
    def getResponseFromGPT(self, pattern):
        try:
            response = self.gpt_client.chat.completions.create(model="gpt-4.1",messages=[{"role":"user","content": pattern},])
        except:
            time.sleep(0.1)
            response = self.gpt_client.chat.completions.create(model="gpt-4.1",messages=[{"role":"user","content": pattern},])

        response = response.choices[0].message.content
        return response
    
    
    ########################### Claude ##############################
    
    def getNERResultStrResponseFomClaude(self, pattern):
        pattern = f"{self.prompt} \
                {self.ner_result_tag_str_with_hint} \
                浮世絵タイトル: '{pattern}'\n \
                質問：「上記の浮世絵タイトルにおいて、地名、演目、替名、役者は何ですか？修正を加えずに、簡潔に一文だけを返事してください。」\n"
        
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

