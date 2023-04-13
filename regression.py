from FB_preprocess import FBPreprocessData
from Linkedin_preprocess import LinkedinPreprocessData
#from Linkedin_preprocess import *
from analysis import model_process
from database import db
import pandas as pd
import json
from flask import request, jsonify, abort, redirect
import flask
from flask_cors import CORS, cross_origin

app = flask.Flask(__name__)
cors = CORS(app)

app.config["DEBUG"] = True
app.config['CORS_HEADERS'] = 'Content-Type'

'''
FB Sample Payload (Fabulous Account)
{"account_id":"1707166029403477","date_range":["2023-03-20","2023-04-09"],"metric":"impressions","locale":"en","ad_ids":["23853845984630413", "23853647368560413", "23853500708100413", "23853472611720413", "23853677032710413", "23853647368560413", "23853677032710413", "23853472611720413", "23853646748110413","23853677032820413", "23853846111380413", "23853845984630413", "23853846111420413"],"platform":"facebook"}
'''
@app.route('/api/regression', methods=['POST'])
def analyze_model():
    raw_json = request.get_json()
    formatted_json = json.dumps(raw_json)
    query_parameters = json.loads(formatted_json)
    metric = query_parameters.get('metric')
    lang = query_parameters.get('lang')
    account_id = query_parameters.get('account_id')
    date_range = query_parameters.get('date_range')
    platform = query_parameters.get('platform')
    ad_ids=query_parameters.get('ad_ids')
    try:
    # Get Filter Ad Ids
        #response_data=LinkedinPreprocessData(account_id,ad_ids,date_range)
        
        '''
        The idea here is to ensure that you have the data in the form of a dataframe df where these columns are a must:
        image_entities : JSON array of image entities
        title_entities : JSON array of title entities
        body_entities : JSON array of body entities
        video_entities: JSON array of video entities
        id: Ad ID / Asset ID whichever is unique
        account_id: Account ID
        All insights should be in new columns : impressions , ctr , clicks , .... etc
        Look at AdsCalculation.csv file for reference
        '''
           
        if(platform=="facebook"):
            df=FBPreprocessData(account_id,ad_ids,date_range)
        elif(platform=="linkedin"):
            df=LinkedinPreprocessData(account_id,ad_ids,date_range) #Need to validate.. do not have data for this platform
            pass
        else:
            pass 

        df["title_entities"]=None #Remove this line if you have title entities coming from Mongo
        df["body_entities"]=None #Remove this line if you have body entities coming from Mongo
        df["account_id"]=account_id
        response_data=model_process(date_range,metric,account_id,lang,df)

        media_analysis = []
        text_analysis = []
        
        for analysis in response_data["analysis"]:
            if(analysis["human_readable"].endswith('video') or analysis["human_readable"].endswith('image') or analysis["human_readable"].endswith('carousel')):
                media_analysis.append(analysis)
            else:
                text_analysis.append(analysis)
        response_data["text_analysis"] = text_analysis
        response_data["media_analysis"] = media_analysis
        del response_data['analysis']
        response={ "data": response_data, "success": True}
        return jsonify(response)
    except Exception as e:
        print("Error inside analysis api",str(e))
        return { "data": {}, "success": False}


if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port=5500)
