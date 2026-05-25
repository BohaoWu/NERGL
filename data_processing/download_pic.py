import json
import config
import requests
import os
import time

def DownloadFileByJsonFile():
    """
    Download ukiyoe pictures by a json file downloaded from label studio.
    """
    count = 0
    with open(config.original_data, "r", encoding='utf-8') as json_file:
        meta_data = json.load(json_file)
    for i in range(len(meta_data)):
        filename = meta_data[i]['data']['url'].split("/", -1)[-1]
        DownloadPictureByUrl(meta_data[i]['data']['url'], folder_path=config.picture_path, filename=filename)
        count+=1
        time.sleep(0.1)

    return


def DownloadPictureByUrl(url, folder_path, filename):

    # 指定下载保存路径
    file_path = os.path.join(folder_path, filename)

    # 创建文件夹（如果不存在）
    os.makedirs(folder_path, exist_ok=True)
    
    headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
    "Referer": "http://www.arc.ritsumei.ac.jp/"  # 很关键
    }

    # 下载文件
    try:
        with requests.get(url, headers=headers, stream=True) as r:
            r.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception:
        print("Error Url is:", url)
    finally:
        pass
    return

def main():
    print("config.picture_path:", config.picture_path)
    DownloadFileByJsonFile()
    return

if __name__ == "__main__":
    main()
