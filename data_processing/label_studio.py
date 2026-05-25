import json
import os
import time
import requests
import urllib

import config

class LabelStudioDataImportTools:
    def __init__(self, inputfile_path, outputfile_path, url_file):
        self.inputfile_path = inputfile_path
        self.outputfile_path = outputfile_path
        self.url_file = url_file
        
        # get url from ukiyo-e
        self.urls = []
        with open(self.url_file, "r", encoding='utf-8') as url_file:
            for url in url_file:
                self.urls.append(url)
        return
    
    def download_image(self, url, save_dir, filename):
        save_path = os.path.join(save_dir)

        try:
            urllib.request.urlretrieve(url, save_dir+filename)
            print(f"✅ Downloaded: {filename}.")
        except Exception as e:
            print(f"❌ Failed to download {url}: {e}.")
        time.sleep(0.5)
    
    def GetTitleById(self, ukiyoe_metadata, id):
        for m in range(len(ukiyoe_metadata)):
            metadata_id = ukiyoe_metadata[m]["data"]["id"]
            if metadata_id == id:
                text = ukiyoe_metadata[m]["data"]["text"]
                return text
        return None

    def GetUrlbyId(self, Id):
        for url in self.urls:
            if Id in url:
                print("Url can not be fount. The id of ukiyoe is ", Id, ".")
                return url
        print("Url can not be fount. The id of ukiyoe is ", Id, ".")
        url = ""
        return url

    def ObjectDetectionBoundBoxJsonTranslater(self):
        '''
        Get metadata which is consisted of filename, ukiyoe title
        '''
        ### read metadata of ukiyoe
        with open(config.original_data, "r", encoding='utf-8') as metadata_file:
            original_file = json.load(metadata_file)
        
        with open(self.inputfile_path, "r", encoding='utf-8') as json_file:
            metadata = json.load(json_file)
        for m in range(len(metadata)):
            ### filename is same as ukiyoe id
            # metadata[m]['data']['filename'] = metadata[m]['data']['url'].split("/", -1)[-1]
            metadata[m]['data']['filename'] = metadata[m]['data']['id']+".jpg"
            # metadata[m]['data']['id'] = metadata[m]['data']['filename'].split(".", 1)[0]
            
            ### add title information into image metadata
            metadata[m]['data']['title'] = self.GetTitleById(ukiyoe_metadata=original_file, id=metadata[m]['data']['id'])
            
            ### add url information if it can be found
            metadata[m]['data']['image'] = "/data/local-files/?d=label-studio/data/upload/Ukiyoe1000/" + metadata[m]['data']['filename']
            # self.download_image(metadata[m]['data']['url'], config.picture_path, metadata[m]['data']['filename'])
            
        with open(self.outputfile_path, "w", encoding="utf-8") as outputfile:
            json.dump(metadata, outputfile, ensure_ascii=False, indent=2)
    
    
def main():
    label_tools = LabelStudioDataImportTools(
        config.original_data,
        config.outputfile_path,
        config.download_url
        )
    label_tools.ObjectDetectionBoundBoxJsonTranslater()    


if __name__ == "__main__":
    main()