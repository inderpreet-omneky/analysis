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

import pandas as pd
from entities_extractor import extractEntities
from database import db
import pymongo
import copy

client = pymongo.MongoClient('mongodb+srv://omneky-dev:yvm0kuPMXSIagh7s@wave.3xxrmwz.mongodb.net/test')
mongodb = client["wave"]

def FBPreprocessData(account_id,ad_ids,date_range):
    response={"data":{}}
    ad_details={}
    df_ad_details={}
    daily_insights_collection=mongodb["daily_insights"]

    for ad in ad_ids:
        df_ad_details[ad]={}
        ad_details[ad]={}
        res=db.table("ads").select("id","ad_type","ad_group_id","landing_page_url","cta","status").where("platform_ad_id",ad).get().first()
        print("The ad is ",ad)
        ad_index=res["id"]
        res2=db.table("ad_groups").select("name","campaign_id").where("id",res["ad_group_id"]).get().first()
        ad_group_name=res2["name"]
        campaign_name=db.table("campaigns").select("name").where("id",res2["campaign_id"]).get().first()["name"]
        ad_details[ad]["adset_name"]=ad_group_name
        ad_details[ad]["campaign_name"]=campaign_name
        ad_details[ad]["type"]=res["ad_type"]
        ad_details[ad]["cta"]=res["cta"]
        ad_details[ad]["status"]=res["status"]
        ad_details[ad]["cta"]=res["cta"]
        if(res.get("landing_page_url")):
            ad_details[ad]["landing_page_url"]=res["landing_page_url"]
        
        
        # Getting daily insights 
        db_filter={'ad_id': ad, 'insight_date': {'$gte': date_range[0], '$lte': date_range[1]}}
        select={"_id":0,"insights":1}
        results=daily_insights_collection.find(db_filter,select)

        insight={"clicks":0,"impressions":0,"spend":0,"ad_pur":0,"total_leads":0,"total_install":0,"cpp":0,"total_reg":0,"total_atc":0,"link_clicks":0,"reach":0}
        insight["spend"]=db.table("daily_insights").select("spend").where("platform_ad_id",ad).where_between("date",date_range).sum("spend")
        if(insight["spend"]==None):
            insight["spend"]=0
        
        length=0
        for result in results:  #Iterate over the date range for that ad
            #result["insights"]=json.loads(result["insights"])
            insight["clicks"]+=int(result["insights"]["clicks"])
            insight["impressions"]+=int(result["insights"]["impressions"])
            insight["cpp"]+=float(result["insights"]["cpp"])  
            insight["reach"]+=int(result["insights"]["reach"])

            try:
                for action in result["insights"]["actions"]:
                    if(action["action_type"]== 'purchase' or action["action_type"]== 'omni_purchase' or action["action_type"]== 'offsite_conversion.fb_pixel_purchase'):
                        insight["ad_pur"]+=float(action["value"])
                    if(action["action_type"]== 'offsite_conversion.fb_pixel_lead' or action["action_type"]== 'lead'):
                        insight["total_leads"]+=float(action["value"])
                    if(action["action_type"]== 'omni_app_install' or action["action_type"]== 'mobile_app_install'):
                        insight["total_install"]+=float(action["value"])
                    if(action["action_type"]== 'app_custom_event.fb_mobile_complete_registration' or action["action_type"]== 'omni_complete_registration'  or action["action_type"]== 'complete_registration'):
                        insight["total_reg"]+=float(action["value"])
                    if(action["action_type"]== 'offsite_conversion.fb_pixel_add_to_cart' or action["action_type"]== 'omni_add_to_cart'  or action["action_type"]== 'add_to_cart'):
                        insight["total_atc"]+=float(action["value"])
                    if(action["action_type"]== 'link_click'):
                        insight["link_clicks"]+=int(action["value"])
                length+=1
            except Exception as e:
                print(str(e))

        ad_details[ad]["cpp"]=round(insight["cpp"]/length,2) if length else 0 #OR ad_spend/total_purchase
        ad_details[ad]["link_clicks"]=insight["link_clicks"]
        ad_details[ad]["reach"]=insight["reach"]
        ad_details[ad]['ctr']=round((insight['clicks']/insight['impressions'])*100,2) if insight['impressions'] else 0
        ad_details[ad]['cpm']=round((insight['spend']/insight['impressions'])*1000,2) if insight['impressions'] else 0
        ad_details[ad]['cpc']=round((insight['spend']/insight['link_clicks']),2) if insight['link_clicks'] else 0
        ad_details[ad]["roas"]=round((insight["ad_pur"]/insight['spend']),2) if insight['spend'] else 0
        #ad_details[ad]["cpr"]=round((insight['spend']/insight['results']),2) if insight['results'] else 0
        ad_details[ad]["cpl"]=round((insight['spend']/insight['total_leads']),2) if insight['total_leads'] else 0
        ad_details[ad]["cpi"]=round((insight['spend']/insight['total_install']),2) if insight['total_install'] else 0
        ad_details[ad]["cpreg"]=round(insight['spend']/insight['total_reg'],2) if insight['total_reg'] else 0
        ad_details[ad]["cpatc"]=round(insight['spend']/insight['total_atc'],2) if insight['total_atc'] else 0
        ad_details[ad]['clicks']=insight['clicks']
        ad_details[ad]['impressions']=insight['impressions']
        ad_details[ad]['spend']=round(insight['spend'],2)

        
        ad_entities=extractEntities(ad_index,ad,account_id)
        ad_details[ad].update(ad_entities)
        
        
        
    df = pd.DataFrame.from_dict(ad_details, orient='index')
    df.reset_index(inplace=True)
    df.rename(columns={"index":"id"},inplace=True)
    # df.to_csv('FBdata.csv', index=False)
    return df


ad_ids=["23853845984630413"]
date_range=["2023-03-20","2023-04-09"]
account_id="1707166029403477"
print("df is",FBPreprocessData(account_id,ad_ids,date_range))