from entities_extractor import extractEntities
from database import db
import pymongo

client = pymongo.MongoClient('mongodb+srv://omneky-dev:yvm0kuPMXSIagh7s@wave.3xxrmwz.mongodb.net/test')
mongodb = client["wave"]

ad_ids=["145119826","146096796","156138906","146096796","156138906","161641376"]

date_range=["2022-12-25","2022-12-30"]
daily_insights_collection=mongodb["daily_insights"]
# Find documents where the 'field' value is in the given list of values
db_filter={'ad_id': {'$in': ad_ids}, 'insight_date': {'$gte': date_range[0], '$lte': date_range[1]}}
select={"_id":0,"insights":1}
results=daily_insights_collection.find(db_filter,select)


#Iterate over the results
# for doc in results:
#     print(doc)

res=db.table("ads").select("platform_ad_id").where("platform_info_id",217).get().all()
data=[i["platform_ad_id"] for i in res]
print(data)