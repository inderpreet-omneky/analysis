'''
Algorithm:
For Every Ad
- Get Ad Id from ads table using platform_ad_id column 
- Get the ad type from ads table (Ad Type)
- Get the landing page url from ads table (Landing Page URL)
- Get the CTA from ads table (CTA)
- Use the platform_ad_id to access daily_insights table and get all other insights (Spend, Clicks,Impressions ,etc)

- Get Image Entities
 --- use Ad Id to get the corresponding platform_asset_id and the source from image table (Source)
 --- use MongoDB to use the platform_asset_id to get image entities from image_entities and image_aws_entities (Image Entities)
 --- use MongoDB to use the platform_asset_id to get image text entities from image_text (Image Text Entities) ---->>>> On Hold as now we are not storing it from importer

- Get Video Entities
 --- use Ad Id to get platform_asset_id and the source from from video table (Source)
 --- use MongoDB to use the platform_asset_id to get video entities from video_entities and video_category_entities and video_aws_entities (Video Entities)
 --- use MongoDB to use the platform_asset_id to get video text entities from video_text_entities (Video Text Entities)
 
 
- Get Text Entities
 --- use Ad Id to get platform_asset_id and text from text table. The text could be either title / body (Title / Body)
 --- use MongoDB to use the platform_asset_id to get text entities from text_entities (Title Entities / Body Entities)

'''


import json

import pandas as pd
from entities_extractor import extractEntities
from database import db
import pymongo

client = pymongo.MongoClient('mongodb+srv://omneky-dev:yvm0kuPMXSIagh7s@wave.3xxrmwz.mongodb.net/test')
mongodb = client["wave"]

def LinkedinPreprocessData(account_id,ad_ids,date_range):
    response={"data":{}}
    ad_details={}
    df_ad_details={}
    daily_insights_collection=mongodb["daily_insights"]

    for ad in ad_ids:
        ad_details[ad]={}
        res=db.table("ads").select("id","ad_type","landing_page_url","cta").where("platform_ad_id",ad).get().first()
        
        ad_index=res["id"]
        
        ad_details[ad]["ad_type"]=res["ad_type"]
        if(res.get("landing_page_url")):
            ad_details[ad]["landing_page_url"]=res["landing_page_url"]
        
        ad_details[ad]["cta"]=res["cta"]
        

        # Getting daily insights 
        db_filter={'ad_id': ad, 'insight_date': {'$gte': date_range[0], '$lte': date_range[1]}}
        select={"_id":0,"insights":1}
        results=daily_insights_collection.find(db_filter,select)

        insight={"clicks":0,"impressions":0,"spend":0,"oneClickLeads":0,"externalWebsiteConversions":0}
        insight["spend"]=db.table("daily_insights").select("spend").where("platform_ad_id",ad).where_between("date",date_range).sum("spend")
        if(insight["spend"]==None):
            insight["spend"]=0
        for result in results:
            #result["insights"]=json.loads(result["insights"])
            print("The insights are ",result["insights"])
            insight["clicks"]+=int(result["insights"]["clicks"])
            insight["impressions"]+=int(result["insights"]["impressions"])
            insight['oneClickLeads']+=(int(result["insights"]["oneClickLeads"]))
            insight['externalWebsiteConversions']+=(int(result["insights"]["externalWebsiteConversions"]))

        ad_details[ad]['ctr']=(insight['clicks']/insight['impressions'])*100 if insight['impressions'] else 0
        ad_details[ad]['cpm']=(insight['spend']/insight['impressions'])*1000 if insight['impressions'] else 0
        ad_details[ad]['cpc']=(insight['spend']/insight['clicks']) if insight['clicks'] else 0
        ad_details[ad]['cost_per_conversions']=insight['spend']/insight['externalWebsiteConversions'] if insight['externalWebsiteConversions'] else 0
        ad_details[ad]['cost_per_lead']=insight['spend']/insight['oneClickLeads'] if insight['oneClickLeads'] else 0
        ad_details[ad]['externalWebsiteConversions']=insight['externalWebsiteConversions'] 
        ad_details[ad]['oneClickLeads']=insight['oneClickLeads']
        ad_details[ad]['clicks']=insight['clicks']
        ad_details[ad]['impressions']=insight['impressions']
        ad_details[ad]['spend']=insight['spend']

        
        df_ad_details[ad]=ad_details[ad]
        ad_entities,df_entities=extractEntities(ad_index,ad,account_id)
        df_ad_details[ad].update(df_entities)
        ad_details[ad].update(ad_entities)
        
        
        
        
    response["data"]["ads"]=ad_details
    response["data"]["Number_of_ads"]=len(ad_details)
    #print(df_ad_details)
    df = pd.DataFrame.from_dict(df_ad_details, orient='index')
    df.to_csv('data.csv', index=False)

account_id=""
date_range=["2023-03-04","2023-04-06"]
ad_ids=['145119826', '145119826', '146096796', '156138906', '161641376', '156138956', '156138956', '143360116', '161641106', '161640996', '143359796', '146097226', '146096626', '161640936', '156138706', '161640996', '161640936', '156138706', '146097226', '143360116', '156138876', '156138876', '146096626', '161641106', '143359796', '161640996', '161640996']
LinkedinPreprocessData(account_id,ad_ids,date_range)