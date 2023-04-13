import ast
import random
import pymongo
client = pymongo.MongoClient('mongodb+srv://omneky-dev:yvm0kuPMXSIagh7s@wave.3xxrmwz.mongodb.net/test')
mongodb = client["wave"]
from database import db
import openai
# openai.api_key = "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"

# def getChatbot(prompt):
#     res=openai.ChatCompletion.create(
#       model="gpt-4",
#       messages=[
#             {"role": "user", "content": prompt}
#         ]
#     )
#     return res["choices"][0]["message"]["content"]




def extractEntities(ad_index,ad,account_id):
    temp={}
    df_entities={}

    image_entities_collection=mongodb["image_entities"]
    image_aws_entities_collection=mongodb["image_aws_entities"]
    video_entities_collection=mongodb["video_entities"]
    video_aws_entities_collection=mongodb["video_aws_entities"]
    video_category_entities_collection=mongodb["video_category_entities"]
    text_entities_collection=mongodb["text_entities"]
    # image_text_collection=mongodb["image_text"]
    video_text_collection=mongodb["video_text"]

    ##-------------------------Image Entities-------------------------##

    '''Storing the image entities from MongoDB''' 
    image_id=db.table("image").select("platform_asset_id").where("ad_id",ad_index).get().first()
    if(image_id):
        image_id=image_id["platform_asset_id"]
        print("The image id is ",image_id)
        try:
            db_filter={'account_id': account_id, 'image_id': image_id}
            select={"entities": 1,"_id":0}
            results=image_entities_collection.find_one(db_filter,select)
            df_entities["image_entities"]=results["entities"]        
            image_entities=[result for result in results["entities"].keys()]        
        except Exception as e:
            print("Failed to get entities because of ",str(e))
            image_entities=[]
            df_entities["image_entities"]={}

        #Getting AWS image entitites from mongo
        try:
            db_filter={'account_id': account_id, 'image_id': image_id}
            select={"entities": 1,"_id":0}
            results=image_aws_entities_collection.find_one(db_filter,select)
            df_entities["image_aws_entities"]=results["entities"]
            image_aws_entities=[result for result in results["entities"].keys()]
        except Exception as e:
            print("Failed to get AWS entities because of ",str(e))
            image_aws_entities=[]

        image_entities=image_entities+image_aws_entities
        df_entities['image_aws_entities']=df_entities.get("image_aws_entities")
        if df_entities["image_aws_entities"]:
            if(image_entities):
                df_entities["image_entities"].update(df_entities["image_aws_entities"])
            else:
                df_entities["image_entities"]={}
        else:
            df_entities["image_aws_entities"]=[]
            image_aws_entities=[]       
        if(df_entities["image_entities"]=={}):
            df_entities.pop("image_entities")


    #-------------------------Video Entities-------------------------#
    
    try:
        video_id = db.table("video").select("platform_asset_id").where("ad_id", ad_index).get().first()
    except Exception as e:
        print("Failed to get video id because of ", str(e))
        video_id = None

    if video_id:
        print("The video id is ", video_id)
        video_id = video_id["platform_asset_id"]

        # Initialize temp dictionary
        temp = {}

        # Function to retrieve video entities from MongoDB
        def retrieve_video_entities(collection, select, temp_key):
            try:
                db_filter = {'account_id': account_id, 'video_id': video_id}
                results = collection.find_one(db_filter, select)
                key_name = list(select.keys())[0]
                temp[temp_key] = results[key_name]
                entities = [result for result in results[key_name].keys()]
            except Exception as e:
                print(f"Failed to get {temp_key} because of ", str(e))
                entities = []
                temp[temp_key] = {}
            return entities

        video_entities = retrieve_video_entities(
            video_entities_collection,
            {"entities": 1, "_id": 0},
            "video_entities"
        )

        video_category_entities = retrieve_video_entities(
            video_category_entities_collection,
            {"category_entities": 1, "_id": 0},
            "video_category_entities"
        )

        video_aws_entities = retrieve_video_entities(
            video_aws_entities_collection,
            {"entities": 1, "_id": 0},
            "video_aws_entities"
        )

        video_entities = list(set(video_entities + video_aws_entities + video_category_entities))
        if video_entities:
            df_entities["video_entities"] = {**temp["video_entities"], **temp["video_category_entities"], **temp["video_aws_entities"]}
        else:
            df_entities["video_entities"] = {}

        if df_entities["video_entities"] == {}:
            df_entities.pop("video_entities")

        try:
            db_filter = {'account_id': account_id, 'video_id': video_id}
            select = {"text": 1, "_id": 0}
            results = video_text_collection.find_one(db_filter, select)
            df_entities["video_text_entities"] = results["text"]
        except Exception as e:
            print("Failed to get video text entities because of ", str(e))
            df_entities["video_text_entities"] = {}

        if df_entities["video_text_entities"] == {}:
            df_entities.pop("video_text_entities")



    ##-------------------------Text Entities-------------------------##

    try:
        ad_text=db.table("text").select("platform_asset_id","type").where("ad_id",ad_index).get().all()
        title_id = None
        body_id = None

        for entry in ad_text:
            if entry['type'] == 'title' and title_id is None:
                title_id = entry['platform_asset_id']
            elif entry['type'] == 'body' and body_id is None:
                body_id = entry['platform_asset_id']
            
            if title_id is not None and body_id is not None:
                break
            
        print("Title id is",title_id)
        print("Body id is",body_id)
    except Exception as e:
        print("Failed to get title / body text from MySQL DB because of ",str(e))
        ad_text=""
    
    try:
        if title_id:
            db_filter = {'account_id': account_id, 'asset_id': title_id}
            select = {'entities': 1, '_id': 0}
            results = text_entities_collection.find_one(db_filter, select)

            if results:
                title_entities = results.get('entities', {})
                if title_entities:
                    df_entities['title_entities'] = title_entities
                    text_entities = list(title_entities.keys())

            if 'title_entities' not in df_entities:
                df_entities['title_entities'] = {}

        if not body_id:
            df_entities.pop('body_entities', None)
        else:
            db_filter = {'account_id': account_id, 'asset_id': body_id}
            select = {'entities': 1, '_id': 0}
            results = text_entities_collection.find_one(db_filter, select)

            if results:
                body_entities = results.get('entities', {})
                if body_entities:
                    df_entities['body_entities'] = body_entities

            if 'body_entities' not in df_entities:
                df_entities['body_entities'] = {}

        if not df_entities:
            print('No title or body entities found for account_id {} and asset_ids {}'.format(account_id, [title_id, body_id]))
        
        return df_entities

    except KeyError as ke:
        print('Error: KeyError while accessing dictionary:', str(ke))

    except TypeError as te:
        print('Error: TypeError while accessing Mongo data:', str(te))

    except Exception as e:
        print('Error: Failed to get title/body entities from Mongo because of', str(e))



    