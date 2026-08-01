import pandas as pd
#from utils.train import get_dataloader
import requests
import json
import time
import os


"""
new_dir = "../foldse"
os.makedirs(new_dir, exist_ok=True)
def fetch_data(iteration: int, data_name: str):
    train_dataloader, val_dataloader, _, num_classes, _ = get_dataloader(
        inference=False, 
        iteration=iteration,
        train_batch=1,
        val_batch=1
    )

    model_path = os.path.join(
        "..", 
        "model-vitsae",
        data_name, 
        "best_topk_model.pth")
    model =  load_vivit_sae_model(
        model_file=model_path, 
    #     model_file2=model_path2, 
    #     model_file3=model_path3, 
    #     model_file4=model_path4,
    #     model_file5=model_path5, 
        num_classes=num_classes
    )
    model.eval()
    with torch.no_grad():
        full_data = []
        for (img, label) in train_dataloader:
            img = img.to('cuda')
            vit_features, sparse_logits, recon_logit, cls_logits = model(img)
            #feat = torch.mean(sparse_logits, dim=1).squeeze(0)
            feat = sparse_logits[:,0].squeeze(0)
            feat = (feat >= 0.5).to(torch.int).detach().cpu()
            row = torch.cat([feat, label.detach().cpu().to(torch.int)], dim=0).tolist()
            full_data.append(row)

        cols = [f"feature_{i+1}" for i in range(feat.size(0))]
        cols.append("label")

        df = pd.DataFrame(full_data, columns=cols)
        for c in cols:
            df[c] = df[c].astype(str)
        df.to_csv(f"{new_dir}/{data_name}_train.csv", index=False)


        full_data = []
        for (img, label) in val_dataloader:
            img, label = img.to('cuda'), label.to('cuda')
            vit_features, sparse_logits, recon_logit, cls_logits = model(img)
            #feat = torch.mean(sparse_logits, dim=1).squeeze(0)
            feat = sparse_logits[:,0].squeeze(0)
            feat = (feat >= 0.5).to(torch.int).detach().cpu()
            _, predicted = torch.max(cls_logits, 1)
            row = torch.cat([feat, predicted.detach().cpu().to(torch.int)], dim=0).tolist()
            full_data.append(row)
    
    
        df = pd.DataFrame(full_data, columns=cols)
        for c in cols:
            df[c] = df[c].astype(str)
        df.to_csv(f"{new_dir}/{data_name}_test.csv", index=False)
"""

result_dir = "foldse_res"
os.makedirs(result_dir, exist_ok=True)
def train_foldse(data_name: str):
    # Enter File Paths
    path = f"foldse/{data_name}_train.csv" # Enter the train file path
    test_path = f"foldse/{data_name}_test.csv" # Uncomment this line if you wish to provide a test dataset

    df = pd.read_csv(path) 
    for c in df.columns:
        df[c] = df[c].astype(str)

    test_df = pd.read_csv(test_path) # Uncomment this line if you wish to provide a test dataset
    for c in test_df.columns:
        test_df[c] = test_df[c].astype(str)

    data_frame_json = df.to_json(orient='split')
    try:
        test_data_frame_json = test_df.to_json(orient='split')
    except:
        test_data_frame_json = ''

    # Set Model Parameters
    numattrs = ''#",".join(df.columns[:-1])#"" # Enter the names of numerical features seperated ',' (For eg - 'feature1,feature2,feature3')
    strattrs = ",".join(df.columns[:-1]) # Enter the names of categorical features seperated by ',' (For eg - 'feature1,feature2,feature3')
    hyp1 = "0.8" # Enter the Train Test split (Ignore if providing a seperate test dataset. If you keep hyp1 as blank or enter a value <0.5 or >=1 only rules will be generated, no testing will be done)
    hyp2 = "1.0"#"0.5" # Enter the Level of exceptions ratio
    hyp3 = "0.0005"#"0.005" # Enter the Tail ratio
    positive_value = "" # Enter the Classification label (Optional and only for binary classification. Keep unchanged otherwise)
    target_column = "label" # Enter the name of the target column
    save_model = ""#"local" # If you want to save the model ("local" will return the json model, else the model won't be saved)
    # Visit http://ec2-52-0-60-249.compute-1.amazonaws.com/example/ to view a example.

    # Enter Your Username and Password
    username = "philiphl2001@yahoo.com" # Enter your registered email id
    password = "hWqeBsrBqa" # Enter your password
    # If your username and password combination does not work, please try to register again with the same email id. Please contact us if that does not work.

    payload = {
        'username': username,
        'password': password,
        'data_frame_json': data_frame_json,
        'numattrs': numattrs,
        'strattrs': strattrs,
        'hyp1': hyp1,
        'hyp2': hyp2,
        'hyp3': hyp3,
        'positive_value': positive_value,
        'test_data_frame_json': test_data_frame_json,
        'label_value': target_column,
        'save_model' : save_model
    }

    #response = requests.post("http://ec2-52-0-60-249.compute-1.amazonaws.com/auth/foldmodel_binary/", json=payload) # Uncomment if you want to run binary classification model.
    response = requests.post("http://ec2-52-0-60-249.compute-1.amazonaws.com/auth/foldmodel_multicategory/", json=payload) # Uncomment if you want to run multi-category classification model.

    try:
        response_obj = response.json()
        try:
            if(response_obj['error']==None):
                # If there is no error the response_obj dictionary contains the following keys - rules, accuracy, f1_score, precision, recall, 
                # n_rules (No of rules), n_preds (No of unique predicates), size (Ruleset size), test_results (List of test dataset results only if test dataset provided).
                # Rules can be accessed with response_obj['rules'] and are returned as a string with two rules seperated by a line break (\n).

                os.makedirs(f"{result_dir}/{data_name}", exist_ok=True)
                with open(f"{result_dir}/{data_name}/full_res.json", "w") as file:
                    json.dump(response_obj, file)

                # Similarly you can use response_obj['accuracy'], response_obj['f1_score'], response_obj['precision'], response_obj['recall'] (These will not be generated if test dataset not provided and (hyp1>=1 or hyp1<0.5 or hyp1 is blank), 
                # response_obj['n_rules'], response_obj['n_preds'], response_obj['size'], response_obj['test_results'] (Only id test dataset provided).
                
                # If save_model = "local" - You can get the model as a json object with response_obj['model_json']. You can save the model to a file using 
                #with open('file_name.json', 'w') as f: f.write(response_obj['model_json'] + '\n')
            else:
                print('Error: ',response_obj['error'])
        except:
            print("There was an error processing your request.")
            #print(response.json())
    except:
        print(response)


datasets = [
    "flowers102",
    #"cars",
    #"fgvc",
    #"caltech101",
    #"foods101",
    #"suns397",
    "dtd",
    "eurosat",
    "pets"
]

for i, data_name in enumerate(datasets):
    #fetch_data(i, data_name)
    print(data_name)
    train_foldse(data_name)
    time.sleep(5)