image_entities=image_entities+image_aws_entities
    if(image_entities):
        df_entities["image_entities"].update(df_entities["image_aws_entities"])
    else:
        df_entities["image_entities"]={}
    if(df_entities["image_entities"]=={}):
        df_entities.pop("image_entities")
image_entities