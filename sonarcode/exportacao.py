import os
import pickle
import pandas as pd
import pandas
import matplotlib.pyplot as plt
import numpy as np
from sonarcode.figuras_merito import sp_index, plota_confusao

class Caminho:
  def __init__(self):
    # self.logs = "/gdrive/MyDrive/lps/goltz/resultados_marlon/logs/"
    # self.modelos =  "/gdrive/MyDrive/lps/goltz/resultados_marlon/modelos/"
    # self.curvas = "/gdrive/MyDrive/lps/goltz/resultados_marlon/curvas/"
    # self.history = "/gdrive/MyDrive/lps/goltz/resultados_marlon/history/"
    # self.his_obj = "/gdrive/MyDrive/lps/goltz/resultados_marlon/history/objeto/"
    # self.his_plan = "/gdrive/MyDrive/lps/goltz/resultados_marlon/history/planilha/"
    # D:\Documents\Python\Article_PassiveSonar
    # d/Documents/Python/Article_PassiveSonar
    self.logs = "/d/Documents/Python/Article_PassiveSonar/logs/"
    self.modelos =  "/d/Documents/Python/Article_PassiveSonar/modelos/"
    self.curvas = "/d/Documents/Python/Article_PassiveSonar/curvas/"
    self.history = "/d/Documents/Python/Article_PassiveSonar/history/"
    self.his_obj = "/d/Documents/Python/Article_PassiveSonar/history/objeto/"
    self.his_plan = "/d/Documents/Python/Article_PassiveSonar/history/planilha/"
  
  def diretorio(self, caminho):
    if os.path.isdir(caminho):
      None
    else:
      os.mkdir(caminho)
    return caminho

class Analise:
  def __init__(self):
    path_init = Caminho()
    self.objeto = path_init.his_obj
    self.planilha = path_init.his_plan

  def save_var(self, historico,_end, nome, test, fold): # salvar objeto do modelo treinado
    with open(self.objeto+'/'+_end+'/'+str(nome) + '_test_' + str(test) + '_fold_' + str(fold), 'wb') as file:
        pickle.dump(historico.history, file)
  
  def save_dados(self, historico,_end, nome, test, fold): # salvar objeto do modelo treinado
    with open(self.objeto+'/'+_end+'/'+str(nome) + '_test_' + str(test) + '_fold_' + str(fold), 'wb') as file:
        pickle.dump(historico, file)
        
  def save_csv(self, data,_end, nome, test, fold): # salvar planilha do modelo treinado
    with open(self.planilha+'/'+_end+'/'+str(nome) + '_test_' + str(test) + '_fold_' + str(fold) + '.csv', 'wb') as file:
      data.to_csv(file.name, encoding='utf-8', index=True)


class Resultados:
  def __init__(self, metrica):
    path_init = Caminho()
    self.objeto = path_config.his_obj
    self.planilha = path_config.his_plan
    self.curvas = path_config.curvas
    self.historico = path_config.history
    self.dir_files = [x for x in os.listdir(self.planilha) if x.startswith(metrica)]

  def csv_individual(self, ntest, nfold):
    teste = "test_" + str(ntest)
    fold = "fold_" + str(nfold)
    files = [x for x in self.dir_files if teste in x]
    data = [x for x in files if fold in x]
    return pandas.read_csv(self.planilha + str(data[0])), data[0]
  
  def csv_lista(self, num_list, teste=True):
    result = []
    if teste:
      num_list = "test_" + str(num_list)
    else:
      num_list = "fold_" + str(num_list)
    files = [x for x in self.dir_files if num_list in x]
    [result.append(pandas.read_csv(self.planilha + str(x))) for x in files]
    return result, files

  def plot_accloss(self, data, nome):
    plt.plot(data.accuracy)
    plt.plot(data.val_accuracy)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(str(self.curvas) + str(nome) + '_acc.png')  
    plt.close()
    plt.plot(data.loss) # summarize history for loss
    plt.plot(data.val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(str(self.curvas) + str(nome) + '_loss.png')
    plt.close()

    
def criar_pastas(path_config, pretrain_config, train_config, lofar_config, 
                 cnn_config, lstm_config, lstm_config2, mlp_config, mlp_config2, dataset_config):
  
  # variaveis iniciais
  # allpasta = [path_config.curvas, path_config.logs, 
  #             path_config.modelos, path_config.his_obj, path_config.his_plan]  
  allpasta = [path_config.curvas, path_config.modelos, path_config.his_obj, path_config.his_plan]
  
  ender_cnn, ender_lstm, ender_lstm_lstm, ender_mlp, ender_mlp_mlp = None, None, None, None, None
  
  # cria estrutura até as janelas
  for pasta in allpasta:

    allender, newender, outender = [], [], []

    if dataset_config.tipo =="normal":
      ender = path_config.diretorio(pasta+'window_'+str(lofar_config.sublofar_size) \
                                    +'_step_'+str(lofar_config.sublofar_step))
    elif dataset_config.tipo =="comite":
      ender = path_config.diretorio(pasta+'comite_window_'+str(lofar_config.sublofar_size) \
                                    +'_step_'+str(lofar_config.sublofar_step))

    # cria estrutura a partir da rede
    if pretrain_config.rede_cnn:
      ender_cnn = ender+'/cnn_'+str(cnn_config.neucnn_1)
      if cnn_config.neucnn_2 != 0:
        ender_cnn = ender_cnn+'_'+str(cnn_config.neucnn_2)  
      if cnn_config.neumlp_1 != 0:
        ender_cnn = ender_cnn+'_mlp_'+str(cnn_config.neumlp_1)
      if cnn_config.neumlp_2 != 0:
        ender_cnn = ender_cnn+'_'+str(cnn_config.neumlp_2)
      allender.append(ender_cnn)

    if pretrain_config.rede_lstm:
      ender_lstm = ender+'/lstm_'+str(lstm_config.neulstm_1)
      if lstm_config.neulstm_2 != 0:
        ender_lstm = ender_lstm+'_'+str(lstm_config.neulstm_2)  
      if lstm_config.neumlp_1 != 0:
        ender_lstm = ender_lstm+'_mlp_'+str(lstm_config.neumlp_1)
      if lstm_config.neumlp_2 != 0:
        ender_lstm = ender_lstm+'_'+str(lstm_config.neumlp_2)
      allender.append(ender_lstm)
    
    if pretrain_config.rede_mlp:
      ender_mlp = ender+'/mlp_'+str(mlp_config.neumlp_1)
      if mlp_config.neumlp_2 != 0:
        ender_mlp = ender_mlp+'_'+str(mlp_config.neumlp_2)  
      allender.append(ender_mlp)
    
    if pretrain_config.rede_lstm_lstm_mlp:
      ender_lstm_lstm = ender+'/lstm_'+str(lstm_config.neulstm_1)
      if lstm_config.neulstm_2 != 0:
        ender_lstm_lstm = ender_lstm_lstm+'_'+str(lstm_config.neulstm_2)  
      if lstm_config.neumlp_1 != 0:
        ender_lstm_lstm = ender_lstm_lstm+'_mlp_'+str(lstm_config.neumlp_1)
      if lstm_config.neumlp_2 != 0:
        ender_lstm_lstm = ender_lstm_lstm+'_'+str(lstm_config.neumlp_2)
      if lstm_config2.neulstm_1 != 0:
        ender_lstm_lstm = ender_lstm_lstm+'_lstm_'+str(lstm_config2.neulstm_1)  
      if lstm_config2.neulstm_2 != 0:
        ender_lstm_lstm = ender_lstm_lstm+'_'+str(lstm_config2.neulstm_2)  
      if lstm_config2.neumlp_1 != 0:
        ender_lstm_lstm = ender_lstm_lstm+'_mlp_'+str(lstm_config2.neumlp_1)
      if lstm_config2.neumlp_2 != 0:
        ender_lstm_lstm = ender_lstm_lstm+'_'+str(lstm_config2.neumlp_2)
      allender.append(ender_lstm_lstm)
      
    if pretrain_config.rede_mlp_mlp:
      ender_mlp_mlp = ender+'/mlp_'+str(mlp_config.neumlp_1)
      if mlp_config.neumlp_2 != 0:
        ender_mlp_mlp = ender_mlp_mlp+'_'+str(mlp_config.neumlp_2)  
      if mlp_config2.neumlp_1 != 0:
        ender_mlp_mlp = ender_mlp_mlp+'_mlp_'+str(mlp_config2.neumlp_1)
      if mlp_config2.neumlp_2 != 0:
        ender_mlp_mlp = ender_mlp_mlp+'_'+str(mlp_config2.neumlp_2)
      allender.append(ender_mlp_mlp)

    # criar os diretórios
    newender = [path_config.diretorio(x) for x in allender]
    if pretrain_config.rede_cnn:
      outender = [path_config.diretorio(x+'/epochs_'+str(train_config.epochs) \
                                        +'_decimate_'+str(lofar_config.decimate)+'_kernel_'+str(cnn_config.kernel_size)) for x in allender]
    else:
      outender = [path_config.diretorio(x+'/epochs_'+str(train_config.epochs) \
                                        +'_decimate_'+str(lofar_config.decimate)) for x in allender]
  
  return [x[x.index('window'):] if dataset_config.tipo =="normal" else x[x.index('comite'):] for x in outender]


def evaluate_CM_and_SP_per_model_and_save(models_list, x, y_true, models_folder):
  # Use a 'models_list', input data 'x', and input label 'y_true' to evaluate the
  # Confusion Matrix (CM) in values (without print), and the Sum-Product (SP)
  # per model in the 'models_list' and save the evaluated arrays containing the
  # CM and SP values in a 'models_folder'
  print_cm = False

  cm_per_model = []
  SP_per_model = []

  models_list_2 = models_list.reshape((10,9))
  for i, (x_fold, y_fold) in enumerate(zip(x, y_true)):
    print(i)
    for model in models_list_2[i]:
      # Fist evaluate CM and SP per model
      cm, y_pred = plota_confusao(lstm_model, x_fold, y_fold, print_cm)
      SP = sp_index(y_fold, y_pred)

      cm_per_model.append(cm)
      SP_per_model.append(SP)

  # Save for Models
  save_name_cm = f'{models_folder}CM_per_model.npz'
  save_name_SP = f'{models_folder}SP_per_model.npz'

  np.savez(save_name_cm, *cm_per_model)
  np.savez(save_name_SP, *SP_per_model)

  return cm_per_model, SP_per_model
