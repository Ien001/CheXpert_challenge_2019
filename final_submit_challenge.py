# encoding: utf-8
import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from util import get_trained_model, get_dataloader

pd.set_option('display.max_colwidth',1000)

def predict_single_model(model, dataloder):
    model.cuda()
    model.eval()
    pred = torch.FloatTensor().cuda()
    with torch.no_grad():
        for i, (inp) in enumerate(dataloder):
            input_var = torch.autograd.Variable(inp.cuda())
            output = model(input_var)
            pred = torch.cat((pred, output.data), 0)

    return pred.cpu().data.numpy()

def ensemble(predictions, ratio):
    prediction_ensemble = np.zeros(shape = predictions[0].shape, dtype = float)
    
    for i in range(0,len(ratio)):
        prediction_ensemble += predictions[i]*ratio[i]

    return prediction_ensemble

def predict_file(prediction_np, input_file, output_file):
    """
    arguments:
        prediction_np:(numpy)   prediction from the model 
        input_file: (csv/txt)   image path list
        output_file:(csv/txt)   out put predicted label as a file

    !!modify to meet your needs!!
    u_one_features = ['Atelectasis', 'Edema']
    u_zero_features = ['Cardiomegaly', 'Consolidation', 'Pleural Effusion']
    """
    def get_final_csv(df):
        result = pd.DataFrame(columns=['Path','Study','Atelectasis','Cardiomegaly','Pleural Effusion','Consolidation','Edema'])
        for Study_name in set(df['Study']):
            tmp = df[df['Study'].isin([Study_name])]
            tmp_result = tmp[0:1].copy()
            tmp_result['Atelectasis'] = tmp['Atelectasis'].max()
            tmp_result['Edema'] = tmp['Edema'].max()
            tmp_result['Cardiomegaly'] = tmp['Cardiomegaly'].mean()
            tmp_result['Pleural Effusion'] = tmp['Pleural Effusion'].mean()
            tmp_result['Consolidation'] = tmp['Consolidation'].mean()

            result = pd.concat([result,tmp_result],axis=0)

        result = result.drop(columns=['Path'])
        return result


    path_df = pd.read_csv(input_file)
    #print(path_df['Path'].str.split('/view'))
    path_df['Study'] = path_df['Path'].str.split('/view',expand=True)[0]
    #print(path_df['Study'])

    pred_df = pd.DataFrame(prediction_np)
    pred_df.columns = ['Atelectasis','Cardiomegaly','Pleural Effusion','Consolidation','Edema']
    #print('pred',pred_df)

    concated_df = pd.concat([path_df,pred_df],axis=1)

    final_result = get_final_csv(concated_df)
    #print('final', final_result)
    final_result.to_csv(output_file, index = False)


def main(argv):
    TEST_IMAGE_LIST = argv[1]
    output_file_path = argv[2]

    cudnn.benchmark = True

    print('loading models & data')
    resumes = ['model_1.pkl',\
            'model_2.pkl',\
            'model_3.pkl',\
            'model_4.pkl',\
            'model_5.pkl',\
            'model_6.pkl']

    weights = [0.2,0.9,0.7,0.4,0.4,0.3]

    # load models & data
    models = []
    dataloders = []
    for resume in resumes:
        model, gray, image_size = get_trained_model(resume.split('_')[0],'Chexpert_challenge_submit/best_models_chexpert_ft/'+resume)
        models.append(model)
        tmp_dataloder = get_dataloader(TEST_IMAGE_LIST,gray,image_size)
        dataloders.append(tmp_dataloder)
    print('load models & data success!')

    print('predicting')
    predictions = []
    for i in range(0,len(resumes)):
        pred_np = predict_single_model(models[i], dataloders[i])
        predictions.append(pred_np)
    print('predict success!')

    print('ensembleing')
    ensemble_result = ensemble(predictions,weights)
    print('ensemble success!') 

    print('predict_file')
    predict_file(ensemble_result, TEST_IMAGE_LIST, output_file_path)
    print('predict_file success!')

if __name__ == '__main__':
    main(sys.argv)
