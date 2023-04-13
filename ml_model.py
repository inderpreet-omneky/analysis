from __future__ import division
import logging
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import shap
import multiprocessing as mp
import time
import requests
import warnings
warnings.filterwarnings("ignore")

class InterpretableModel:

    def __init__(self, data_obj, metric, account_id,lang_flag,num_feats=10):

        self._metric = metric
        self._data_obj = data_obj
        self._account_id = account_id
        self._ads_used = None
        self._b_feats_used = None
        self._s_feats_used = None
        self.coeffs = None
        self.testlist = None
        self.savedollar = None
        self.confidence = None
        self._lang_flag = lang_flag

    def interpreting_model(self,data_type):
        return self._random_forest_pipline(self._metric, self._account_id,data_type, num_feats=10)

    def summary_plot(self,shap_values,data_type,features=None, feature_names=None, max_display=None, sort=True, importance = None):
        min_percent_impact = []
        max_percent_impact = []
        ad_id = []
        def get_max(row):
            if abs(row['min_percent_impact']) > abs(row['max_percent_impact']):
                return row['min_percent_impact']
            else:
                return row['max_percent_impact']
        # print("I am in analysis.py 2------------------------------------------------------->")
        multi_class = False
        if isinstance(shap_values, list):
            multi_class = True
            plot_type = "bar" # only type supported for now
        else:
            assert len(shap_values.shape) != 1, "Summary plots need a matrix of shap_values, not a vector."
        # convert from a DataFrame or other types
        if str(type(features)) == "<class 'pandas.core.frame.DataFrame'>":
            # print("I am in analysis.py 3 ===============================------------------------------------------------------->")
            if feature_names is None:
                # feature_names = features.columns
                features.columns=[regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in features.columns.values]
                feature_names = features.columns
                # print("In the analysis-------------------------------------------------------------------------------------",feature_names)
            dataframe = features
            # print("In the analysis 1.1    -------------------------------------------------------------------------------------",dataframe)
            features = features[feature_names].values
            # print("In the analysis 1.2    -------------------------------------------------------------------------------------",features)
        elif isinstance(features, list):
            if feature_names is None:
                # print("In the analysis 1.3    -------------------------------------------------------------------------------------",features)
                feature_names = features
            features = None
        elif (features is not None) and len(features.shape) == 1 and feature_names is None:
            # print("In the analysis 1.41       -------------------------------------------------------------------------------------",feature_names)
            feature_names = features
            features = None

        num_features = (shap_values[0].shape[1] if multi_class else shap_values.shape[1])

        if feature_names is None:
            feature_names = np.array([labels['FEATURE'] % str(i) for i in range(num_features)])

        if max_display is None:
            max_display = features.shape[1]
        if sort:
            # order features by the sum of their effect magnitudes
            if multi_class:
                feature_order = np.argsort(np.sum(np.mean(np.abs(shap_values), axis=0), axis=0))
            else:
                feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
                feature_confidence = np.sum(np.abs(shap_values), axis=0)
            feature_order = feature_order[-min(max_display, len(feature_order)):]
        else:
            feature_order = np.flip(np.arange(min(max_display, num_features)), 0)
        row_height = 0.4
        for pos, i in enumerate(feature_order):
            shaps = shap_values[:, i]
            values = None if features is None else features[:, i]
            inds = np.arange(len(shaps))
            np.random.shuffle(inds)
            if values is not None:
                values = values[inds]
            shaps = shaps[inds]
            #print("shaps:",shaps)
            colored_feature = True
            try:
                values = np.array(values, dtype=np.float64)  # make sure this can be numeric
            except:
                colored_feature = False
            N = len(shaps)
            nbins = 100
            quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8))
            inds = np.argsort(quant + np.random.randn(N) * 1e-6)
            layer = 0
            last_bin = -1
            ys = np.zeros(N)
            for ind in inds:
                if quant[ind] != last_bin:
                    layer = 0
                ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
                layer += 1
                last_bin = quant[ind]
            ys *= 0.9 * (row_height / np.max(ys + 1))
            if features is not None:
                assert features.shape[0] == len(shaps), "Feature and SHAP matrices must have the same number of rows!"

                #nan_mask = np.isnan(values)
                nan_mask  = (values != 1) | np.isnan(values)
                min_percent_impact.append(np.mean(shaps[np.invert(nan_mask)][shaps[np.invert(nan_mask)]<0]))
                max_percent_impact.append(np.mean(shaps[np.invert(nan_mask)][shaps[np.invert(nan_mask)]>=0]))
        df = pd.DataFrame({"feature_name":[feature_names[i] for i in feature_order],'confidence':[importance[feature_names[i]] for i in feature_order],
                            "min_percent_impact":min_percent_impact,"max_percent_impact":max_percent_impact,"ad_id":[self.get_ad_id(dataframe, feature_names[i]) for i in feature_order]})
        #print(df)
        #print(df[''])
        df = df.sort_values(by = 'confidence')
        #print(df)
        d_f = df.iloc[::-1]
        d_f = d_f.fillna(0)
        #print(d_f)
        d_f['max_impact'] = d_f.apply(get_max, axis = 1)
        d_f = d_f[d_f['max_impact'] != 0]
        #print(d_f)
        #print(d_f['max_impact'])
        #print(d_f[['feature_name','max_impact']])
        #print(d_f['feature_name'])
        
        print("Exited Summary Plot")
        return d_f[['feature_name','max_impact','confidence','ad_id']]

    def _random_forest_pipline(self,metric, account_id,data_type, num_feats=10):
        tm = time.time()
        print("inside line------------------165")
        dftemp, dummies, xvars = self._filter_df()
        print("after filtering----------------------",dftemp)
        print("find filter time","*"*10,time.time()-tm)
        dataframe = dftemp
        # print("In the random forest pipeline-------------------------------------------------------------------------------------------",dftemp.columns)
        assert metric in ['cpp','ctr','cpc','impressions','cpr',"cpi","cpl","cpreg","cpatc",'cost_per_conversions','cost_per_lead','cpm']
        # print("Below therandom forest pipeline 1 -------------------------------------------------------------------------------------------------------------------------------------------------------------------",dftemp[feats])
        print("line are 173-------------------------",dataframe)
        feats = list((dummies|xvars)) 
        # print("Below the random forest pipeline 1.1-----------------------------------------------------------------------------------------------------------------------------------------------------------------",feats)
        spend = np.mean(dftemp['spend'])
        clicks = np.mean(dftemp['clicks'])
        print("line are 179-------------------------",spend,"---",clicks)

        # print("Below the random forest pipeline 1.2-------------------------------------------------------------------------------------------------------------------------------------------------------------------",spend)
        impression = np.mean(dftemp['impressions'])
        print("line are 182-------------------------",impression)

        if metric=='impressions':
            dftemp = dftemp[[metric] + feats]
            print("inside 186------",dftemp)
            # print("Below the random forest pipeline 1.3-------------------------------------------------------------------------------------------------------------------------------------------------------------------",dftemp)
        else:ars = ['nlp_link_description_sentiment_mag_mean',
         'nlp_link_description_sentiment_score_mean',
         'nlp_link_description_documentSentiment_mag',
         'nlp_link_description_documentSentiment_score',
         'nlp_title_sentiment_mag_mean',
         'nlp_title_sentiment_score_mean',
         'nlp_title_documentSentiment_mag',
         'nlp_title_documentSentiment_score',
         'nlp_body_sentiment_mag_mean',
         'nlp_body_sentiment_score_mean',
         'nlp_body_documentSentiment_mag',
         'nlp_body_documentSentiment_score',
         'nlp_video_transcription_sentiment_mag_mean',
         'nlp_video_transcription_sentiment_score_mean',
         'nlp_video_transcription_documentSentiment_mag',
         'nlp_video_transcription_documentSentiment_score','VID_segment_time']
        # Exclude binary features not found in the dataframe')
        # print("Below the ml model 1 -------------------------------------------------------------------------------------------------------------------------------------------------------------------",dftemp.columns)
        dftemp.dropna(inplace=True, axis=0)

        self._ads_used, self._b_feats_used, self._s_feats_used = dftemp.index.tolist(), dummies,xvars
        logging.info("# of matching ads: {}".format(len(dftemp)))
        # print("inside 213------",self._b_feats_used,)

        logging.info("# of dataframe columns: {}".format(len(list(feats))))
        rf = RandomForestRegressor(random_state = 0,n_jobs = -1)
        # print("inside line 211------",rf)
        rf.fit(dftemp[feats].values, dftemp[metric].values)
        dftttmp = dftemp[feats]
        # print("inside line 220------",dftttmp)
        dftttmp['ctr'] = dftemp[metric].values
        # print("inside line 222------",dftttmp['ctr'])
        Score = rf.score(dftemp[feats].values, dftemp[metric].values)
        calculate_percentage = float(np.average(rf.predict(dftemp[feats].values)))
        feature_importance = dict(zip(feats,rf.feature_importances_))
        # print("inside line 225------",feature_importance)

        shap_values = shap.TreeExplainer(rf).shap_values(dftemp[feats])
        # print("inside lin 229---------------",shap_values)
        #shap.summary_plot(shap_values,dftemp[feats])
        dffinal = self.summary_plot(shap_values,data_type,features = dataframe,feature_names = feats, importance = feature_importance)
        # print("inside lin 231---------------",dffinal)

        if len(dffinal)==0:
            raise Exception("Not enough data to analyze!")
        #print("$"*100,"dffinal['max_impact']")
        #print(dffinal['max_impact'])
        dffinal['percent_impact'] = (dffinal['max_impact']/calculate_percentage)* 100
        # print("before that the coeff 236",dffinal)

        if metric.lower() == 'ctr':
            dffinal['save_dollar'] = dffinal['max_impact'] * spend
            self.save_dollar = list(dffinal['save_dollar'])
        elif metric.lower() == 'cpc':
            dffinal['save_dollar'] = dffinal['max_impact'] *len(dffinal['max_impact'])
            self.save_dollar = list(dffinal['save_dollar'])
        elif metric.lower() == 'cpm':
            dffinal['save_dollar'] = dffinal['max_impact'] * (impression/1000)
            self.save_dollar = list(dffinal['save_dollar'])
        elif metric.lower() == 'cpp':
            dffinal['save_dollar'] = ['None'] * len(dffinal['max_impact'])
            self.save_dollar = list(dffinal['save_dollar'])
        elif metric.lower() == 'cpr':
            dffinal['save_dollar'] = ['None'] * len(dffinal['max_impact'])
            self.save_dollar = list(dffinal['save_dollar'])
        elif metric.lower() == 'cost_per_lead':
            dffinal['save_dollar'] = ['None'] * len(dffinal['max_impact'])
            self.save_dollar = list(dffinal['save_dollar'])
        elif metric.lower() == 'cpreg':
            dffinal['save_dollar'] = ['None'] * len(dffinal['max_impact'])
            self.save_dollar = list(dffinal['save_dollar'])
        elif metric.lower() == 'cpatc':
            dffinal['save_dollar'] = ['None'] * len(dffinal['max_impact'])
            self.save_dollar = list(dffinal['save_dollar'])
        elif metric.lower() == 'cpi':
            dffinal['save_dollar'] = ['None'] * len(dffinal['max_impact'])
            self.save_dollar = list(dffinal['save_dollar'])
        elif metric.lower() == 'cost_per_conversions':
            dffinal['save_dollar'] = ['None'] * len(dffinal['max_impact'])
            self.save_dollar = list(dffinal['save_dollar'])
        elif metric.lower() == 'impressions':
            dffinal['save_dollar'] = ['None'] * len(dffinal['max_impact'])
            self.save_dollar = list(dffinal['save_dollar'])
        self.coeffs = list(dffinal['percent_impact'])#coeffs
        self.testlist = list(dffinal['feature_name'])#df['feats'][:n_feats]
        self.confidence = list(dffinal['confidence'])
        # print("before that the coeff 272",self.coeffs)
        #print(dffinal['feature_name']) 
        # print("LET's SEE NOW!")
        # print("in analysis line 260---------------")
        dffinal['feature_name'] = dffinal['feature_name'].apply(self.make_readable)
        return len(dftemp),Score, dffinal



    def _filter_df(self):
        df = self._data_obj.df
        dummies = self._data_obj.binary_feats
        xvars = self._data_obj.scalar_feats
        # print("the df in filter --------------------------------------------------",df.columns)
        # print("the dummies in filter-----------------------------------------------------",dummies)
        # print("the vars in filter----------------------------------------------------------",vars)
        dummies_low_var = set()
        dummies_many_null = set()
        xvars_many_null = set()
        df_tmp  =pd.DataFrame(1 - df[list(dummies)].sum(axis = 0)/len(df), columns = ['percent'])
        # print("Inside the filter_df---------------------------------------------------------------------------------------------",df_temp.columns)
        dummies_low_var = set(list(df_tmp[df_tmp['percent']>.96].index))
        temp_series = 1 - df[list(dummies)].isna().sum()/len(df)
        dummies_many_null = set(list(temp_series[temp_series<.99].index))
        sparse_vars = dummies_low_var | dummies_many_null | xvars_many_null
        logging.info('Remaining sparse features: {}'.format(sparse_vars))
        df = df.drop(sparse_vars, axis=1)

        # Filter down dummies and xvars
        binary_feats = dummies  - sparse_vars
        # print("+"*100,"Binary feats:")
        # print(binary_feats)
        scalar_feats = xvars - sparse_vars
        return df, binary_feats, scalar_feats

    def get_salients(self,df,data_type):
        omit_list = ['vision_entities_font', 'vid_all_entities_font']
        df_used = self._data_obj.df.loc[self._ads_used]
        analysis = []
        NUM_ADS_PER_FEAT = 20
        #print("!"*100)
        #print(self.coeffs)
        TRUNCATE_VALUE = max(self.coeffs)
        unique_keyword=set()
        unique_keyword.add("Ad type is video")
        anlysis_lstid = []
        flag1=False
        #print("READABLE NAMES:")
        for feat, coeff, dollar, conf in zip(self.testlist, self.coeffs, self.save_dollar,self.confidence):
            if feat not in omit_list:
                # print(" condition 2 in salient featues ---------------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",feat)
                coeff = max(-TRUNCATE_VALUE, min(coeff, TRUNCATE_VALUE))
                if coeff>0 and coeff<0.05:
                    continue
                if coeff<0 and coeff>-0.05:
                    continue
                flag1=False
                l1=feat.split()
                for j in range(len(l1)):
                    if len(l1[j])<4:
                        flag1=True
                    continue
                # print(" condition 3----------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",coeff)
                if flag1==False:
                    readable_name = self.make_readable(feat)
                else:
                    continue
                if readable_name.find("is")!=-1:
                    continue
                # print(" condition 4 in salient featues ---------------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",readable_name)
                # print(" condition 5 in salient featues ---------------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",type(readable_name))

                readable_name = readable_name.rstrip()
                readable_name = readable_name.lstrip()
                readable_name = re.sub(r'^https?:\/\/.*[\r\n]*','', readable_name)
                readable_name = re.sub(r'http\S+','',readable_name)
                readable_name = re.sub(r'\.+', ".", readable_name)
                readable_name=readable_name.replace("\\","")
                # text_tokens = word_tokenize(readable_name)
                # tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
                # filtered_sentence = (" ").join(tokens_without_sw)
                # readable_name=filtered_sentence
                if readable_name in unique_keyword:
                    continue
                else:
                    unique_keyword.add(readable_name)
                # print(" condition 7----------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",readable_name)
                #print(ascii(readable_name))    
                feat_dict = {'name': feat, 'coeff': coeff, 'ads': [], 'human_readable': readable_name, "dollar_info":dollar, "confidence":conf}  
                # Get used rows that have the feature:
                # TODO: Weight with impressions?
                if feat in self._b_feats_used:
                    df_feat = df_used.loc[df_used[feat] == 1]
                    df_feat = df_feat.sort_values(self._metric, ascending=False)
                    feat_dict['ads'] = df_feat.head(NUM_ADS_PER_FEAT)['id'].tolist()
                    anlysis_lstid.append(df_feat.head(NUM_ADS_PER_FEAT)['id'].tolist())         
                elif feat in self._s_feats_used:
                    # TODO: Deal with scalars properly (still want good performers)
                    df_feat = df_used.sort_values(feat, ascending=False)
                    feat_dict['ads'] = df_feat.head(NUM_ADS_PER_FEAT)['id'].tolist()
                    anlysis_lstid.append(df_feat.head(NUM_ADS_PER_FEAT)['id'].tolist())        
                else:
                    raise ValueError('Feature {} is neither binary nor scalar'.format(feat))         
                analysis.append(feat_dict)
            else:
                print("Omitted feat ", feat)
        print("data type in line 362",data_type)
        lst=df.columns.tolist()
        lst=[x for x in lst if 'entities_' not in x and "video_text_entities" not in x]
        if data_type=='image':
            print("inside analysis data typ 363")
            lst=df.columns.tolist()
            print("The list is ",lst)
        #     lst=[x for x in lst if 'entities_' not in x]
        #     #lst=['title','title_entities','account_id','spend','clicks','impressions','ctr','ad_type','id','source',self._metric,'cpc','cost_per_lead','cpm','image_entities','body_entities','externalWebsiteConversions','cost_per_conversions','cost_per_lead','body','cta']
        # if data_type=='video':
        #     lst=df.columns.tolist()
        #     print("The list is ",lst)
        #     lst=[x for x in lst if 'entities_' not in x]
        #     #lst=['title','video_entities','title_entities','account_id','spend','clicks','impressions','ctr','ad_type','id','source',self._metric,'cpc','cost_per_lead','cpm','body_entities','externalWebsiteConversions','cost_per_conversions','cost_per_lead','body','cta']
        # if data_type=='both':
        #     lst=df.columns.tolist()
        #     lst=[x for x in lst if 'entities_' not in x]
        #     #lst=['title','video_entities','title_entities','account_id','spend','clicks','impressions','ctr','ad_type','id','source',self._metric,'cpc','cost_per_lead','cpm','body_entities','externalWebsiteConversions','cost_per_conversions','cost_per_lead','body','image_entities','cta']
        df=self._data_obj.df[lst].rename(columns={self._metric:"metric"}).set_index('id')
        df.reset_index(inplace=True)
        return analysis,df.set_index('id').to_dict('index')      

    #To make human readable forms
    def make_readable(self, orig_name): 
        #print("INSIDE make_readable")
        language_str = self._lang_flag
        # print(" condition 4 in make_readable----------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",language_str)
        # print("orig_name:")
        # print(orig_name)
        name =  orig_name.strip().replace("-", "_").split('_')
        # print(" condition 5 in make_readable----------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",name)
        while name[-1] == "":
            name = name[:-1]
        # print(" condition 6 in make_readable----------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",name)
        #handling these three type of variables for now, for more variable types, these rules will be amended.
        if (name[0]).lower() == 'video':
            if (name[1]).lower() == 'text':
                final_name = ('"' + name[-1].lower() + '"' + ' Text in Video'.lower())
                # print("In the analysis.py vid---------- ---------------------=================================>",final_name)
            elif language_str:
                # print(" condition 7. in make_readable----------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",language_str)
                if language_str!='en':
                    translated_name = self.translate_text(text=name[-1].lower(),target_language_string=language_str)
                    final_name = '"'+ translated_name  +'"'+ ' in Video'.lower()
                else:
                    final_name ='"'+  name[-1].lower() +'"'+ ' in Video'.lower()
            else:
                final_name =  '"'+ name[-1].lower() +'"'+ ' in Video'.lower()
        elif (name[0]).lower() == 'title' :
                final_name = ('"' + name[-1].lower() + '"' + ' in title'.lower())
        elif (name[0]).lower() == 'body' :
                final_name = ('"' + name[-1].lower() + '"' + ' in body'.lower())
        elif name[0].lower() == 'type':
            final_name = ('Ad type is ' + '"' + name[-1].lower() + '"')
        elif (name[0]).lower()  == 'image':
            if 'image_api_full_text' in orig_name:
                final_name =  '"' + name[-1].lower() + '"' + ' in Image'.lower()
            else:
                if language_str:
                    if language_str!='en':
                        translated_name = self.translate_text(text=name[-1].lower(),target_language_string=language_str)
                        final_name = translated_name  + ' in Image'.lower()
                    else:
                        final_name = '"'+name[-1].lower() + '"'+ ' in Image'.lower()
                else:
                    final_name = '"'+name[-1].lower() + '"'+ ' in Image'.lower()
        elif (name[1] == 'color'):
            if ('primary' in name):
                if language_str:
                    if language_str!='en':
                        translated_name = self.translate_text(text='Primary color is ' + name[-1].lower(),target_language_string=language_str)
                        final_name =  translated_name + ' in Image'.lower()
                    else:                
                        final_name =  'Primary color is ' + name[-1].lower() + ' in Image'.lower()
                else:                
                    final_name =  'Primary color is ' + name[-1].lower() + ' in Image'.lower()
            elif ('secondary' in name):
                if language_str:
                    if language_str!='en':
                        translated_name = self.translate_text(text='Secondary color is ' + name[-1].lower(),target_language_string=language_str)
                        final_name = '"' + translated_name + '"' + 'in Image'.lower()
                    else:
                        final_name =  'Secondary color is ' + name[-1].lower() + ' in Image'.lower()
                else:
                    final_name =  'Secondary color is ' + name[-1].lower() + ' in Image'.lower()
            elif ('tertiary' in name):
                if language_str:
                    if language_str!='en':
                        translated_name = self.translate_text(text='Tertiary color is ' + name[-1].lower(),target_language_string=language_str)
                        final_name = '"'+translated_name + '"'+ 'in Image'.lower()
                    else:
                        final_name =  'Tertiary color is ' + name[-1].lower() + ' in Image'.lower()
                else:
                    final_name =  'Tertiary color is ' + name[-1].lower() + ' in Image'.lower()
        else:
            final_name = orig_name
        # print(final_name)
        return final_name

    def order_format_dataframe(self,dct):
        def get_order(x):
            if 'video' in x.lower():
                return 0
            elif 'image' in x.lower():
                return 1
            elif 'headline' in x.lower():
                return 2
            elif 'body text' in x.lower():
                return 3
            elif 'link description' in x.lower():
                return 4
            else:
                return 5
        dct['feature_sort'] = dct['feature_name'].apply(lambda x: get_order(x))
        dct = dct.sort_values(by=['feature_sort'])
        dct = dct.drop(['feature_sort'], axis=1)
        return dct

    def get_ad_id(self,df_used, feat):
        df_feat = df_used.loc[df_used[feat] == 1]
        #df_feat = df_feat.sort_values(self._metric, ascending=False)
        return df_feat['id'].tolist()
    
    
    def translate_text(self, text, target_language_string):
        """Translating Text."""
        # print("Translating Text")
        
        json_data = {
                        'q': text,
                        'source': 'en',
                        'target': target_language_string,
                        'format': 'text'
                    }
        try:
            req = requests.post('https://translation.googleapis.com/language/translate/v2?key=AIzaSyBSSP5BNxjVUOFtP54jrGcPnk2WOkLEiSo', json=json_data)
            response = req.json()
            #print(dict(response)['data']['translations'][0]['translatedText'])
            translated_text = dict(response)['data']['translations'][0]['translatedText']
        except:
            translated_text = text
        return translated_text



