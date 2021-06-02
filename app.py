import os
#from zipfile import ZipFile
import ast
import os 
import collections
import numpy as np
import pandas as pd

from textblob import TextBlob
from matplotlib.ticker import StrMethodFormatter

import texthero as hero

import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')

from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import json 

import collections
from tqdm import tqdm


from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from pyngrok import ngrok
import nest_asyncio
import uvicorn
import os 

#=====================================================#

API_KEY = "NEURALPOCKET786#%*@"

#=====================================================#



print("All packages imported !!!")


class processReviews(BaseModel):
    api_key: str
    json_input_data: dict
    features:list

app = FastAPI()


@app.post("/processCSV/")
async def takeQue(payload: processReviews):

  payload_dict = payload.dict()



  # Where processed csv will store
  json_input_data = payload_dict['json_input_data']
  features = payload_dict['features']
  api_key = payload_dict['api_key']

  if API_KEY == str(api_key):
        
      print("[payload_dict] :",payload_dict)


      # Beacuse list is in string '[]'  
      #features = ast.literal_eval(features)

      print("[json_input_data] :",type(json_input_data))

      print("[features] :",features,type(features))




      print("\n PROCESSING DATA")
      print("-"*150)




      reviewsRaw = pd.DataFrame.from_dict(json_input_data)

      print("[Shape] : ",reviewsRaw.shape)

      reviewsRaw = reviewsRaw.dropna()
      Shape_of_data = reviewsRaw.shape

      reviewsRaw['combine'] = reviewsRaw['title'] + " " + reviewsRaw['review_text']

      #Clean text
      reviewsRaw['clean'] = hero.clean(reviewsRaw['combine'])

      all_f_found = []
      all_records = []


      def finalScore(score):

        if score >= 0.05 :
            return "posstive"

        elif score <= - 0.05 :
            return "negative"

        else :
            return "neutral"


      for index, row in tqdm(reviewsRaw.iterrows()):
        sentence = row['combine'].lower()

        tokenized_sentence = nltk.word_tokenize(sentence)

        sid = SentimentIntensityAnalyzer()


        f = features
        f = [item.lower() for item in f]


        for i in f:

          if i in sentence:


            all_f_found.append(i)

            sentences = [sentence + '.' for sentence in sentence.split('.') if i in sentence]

            score = sid.polarity_scores(sentence)




            current_record = {"found":i,
                              "sentence":sentence,
                              "positive":score['pos'],
                              "negative":score['neg'],
                              "neutral":score['neu'],
                              "overall_sentiments":finalScore(score['compound']),
                              "score":score['compound']*100}

            all_records.append(current_record)

            print(f'\n [Found] :', current_record )


      final = pd.DataFrame(all_records)
      print("[OUTPUTFILE SHAPE] :",final.shape)


      return {"Status":"DONE","ALL_JSON":final.to_json(orient="records")}


  else:
      return {"Status":"Invalid API Key"}
    



############################################################################################

# DRIVER CODE

# if __name__ == "__main__":
# 	print("STARTING API...")
# 	uvicorn.run(app, host="0.0.0.0",port=8000)


############################################################################################
