import numpy as np
import multiprocessing as mp
from ml_model import InterpretableModel

from data_process import Data

def model_process(date_range,metric,account_id,lang,df,call="analysis"):
    lang_flag = lang
    try:
        if len(df)>0:
            # print("inside here the df")
            # print("colums are-----------------------",df.columns)
            data_obj,df=multiprocess(df)
            # print("df after multiprocess--------------------------",df.shape)
            # print("the columns are ------------------------",df.columns)
            print("Executed multiprocess--------------------")
            if 'video_entities' in df.columns and 'image_entities' in df.columns:
                flag='both'
            elif 'image_entities' in df.columns:
                flag='image'
            elif 'video_entities' in df.columns:
                flag='video'
                print("inside 3")
            # print("the flag is---------------------------",flag)
            if call=="irrelevant_feats":
                pass
                # try:
                #     print("Mission - Store Irrelevant Feats!!")
                #     start_date=date_range[0]
                #     end_date=date_range[1]
                #     model = InterpretModel(data_obj, metric.lower(), account_id,lang_flag,start_date,end_date) #model for irrelevant feats
                #     response = model.getIrrelevantFeats(flag) 
                #     response = {"result":response[0],'status': response[1]}
                #     return(response) 
                # except Exception as e:
                #     response=f"Issue while  storing irrelevant feats {str(e)}"
                #     return(response)
            else:
                model = InterpretableModel(data_obj, metric.lower(), account_id,lang_flag) #model for analysis
                No_of_ads, r2_score, df_fnl = model.interpreting_model(flag) 
            
            print("the ads are------------------------",No_of_ads)
            salient_feat, analysis_list_id = model.get_salients(df,flag)
            print("in line 45---------------------data_processing")
            results = { "metric": metric }
            results['analysis']=salient_feat
            results['Number_of_ads'] = No_of_ads
            results['r2_score'] = r2_score
            results['ads'] = analysis_list_id
            return results
    except Exception as e :
        print("no data  in data process !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! becuase of ",str(e))
        return None

def multiprocess(df):
    if len(df) > 500:
        chunk = 100
    else:
        chunk = 5
    # print("before line 66----------------------",chunk)
    
    df_lst = [i for i in np.array_split(df, chunk)]
    pool = mp.Pool(mp.cpu_count() - 1)# always the processors should be greater than 1
    result=pool.imap(Data, df_lst)
    pool.close()
    pool.join()
    result = list(result)
    data_obj = result[0]
    temp_df, temp_scalar_feat, temp_binary_feat = result[0]._df, result[0]._scalar_feats, result[0]._binary_feats
    for i in result[1:]:
        try:
            temp_df = temp_df.append(i._df, sort=False)
        except:
            pass
        try:
            temp_scalar_feat.update(i._scalar_feats)
        except:
            pass
        try:
            temp_binary_feat.update(i._binary_feats)
        except:
            pass
        
    temp_df = temp_df.fillna(0)
    temp_df = temp_df.reset_index(drop=True)
    data_obj._df = temp_df
    data_obj._scalar_feats = temp_scalar_feat
    data_obj._binary_feats = temp_binary_feat
    df = data_obj.df
    df_without_filter = df.copy()
    df_without_filter = df_without_filter.reset_index()
    df.rename(columns={"ad_id":"id"},inplace=True)
    return data_obj,df