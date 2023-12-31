import numpy as np
import seaborn as sns
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
import matplotlib.pyplot as plt
import math

# indice sp
def sp_index(y_true, y_pred):
    num_classes = len(np.unique(y_true))
    recall = recall_score(y_true, y_pred, average=None)
    sp = np.sqrt(np.sum(recall) / num_classes *
                 np.power(np.prod(recall), 1.0 / float(num_classes)))
    return sp
  

# plotar matriz confusão
def plota_confusao(model,x, y, print):
  y_pred_prob = model.predict(x)
  y_pred = y_pred_prob.argmax(axis=-1)
  confusao = confusion_matrix(y, y_pred)

  if print == 1:
    fig, ax = plt.subplots()
    sns.heatmap(confusao, annot=True, ax=ax, fmt='d', cmap='Reds')
    ax.set_title("Confusion Matrix", fontsize=18)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
  return confusao, y_pred

def plot_confusion_matrix(cm, cms,  classes, normalizar,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalizar:
        cm = cm.astype(float)
        cms = cms.astype(float)
        ncm = cm
        ncms = ncms
        lines_sum = cm.sum(axis=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ncm[i,j] = cm[i,j] / lines_sum[i]
                ncms[i,j] = cms[i,j] / lines_sum[i]
        ncm = 100*ncm
        ncms = 100*ncms
    else:
        ncm = cm
        ncms = cms
                              
    plt.imshow(ncm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize = 20)
    plt.yticks(tick_marks, classes, fontsize = 20)

    thresh = ncm.max() / 2.

                              
    for i in range(ncm.shape[0]):
        for j in range(ncm.shape[1]):
            if normalizar:
                print_str = '{0:.2f}%'.format(ncm[i, j]) + '\n$\pm$' + '{0:.2f}%'.format(ncms[i, j])
            else:
                print_str = '{0:.1f}'.format(ncm[i, j]) + '\n$\pm$' + '{0:.1f}'.format(ncms[i, j])
                
            plt.text(j, i, print_str,
                     horizontalalignment="center",
                     verticalalignment="center", fontsize=15,
                     color="white" if ncm[i, j] > thresh else "black")
            
    plt.tight_layout()

    # plt.ylabel('Classe Verdadeira', fontsize = 25)
    # plt.xlabel('Classe Predita', fontsize = 25)
    plt.ylabel('True Class', fontsize = 25)
    plt.xlabel('Predicted Class', fontsize = 25)
  

# Curva de validação
def curvalearnvalid(train_history, path):
  # list all data in history
  # print(train_history.history.keys())
  # summarize history for accuracy
  plt.plot(train_history.history['accuracy'])
  plt.plot(train_history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'valid'], loc='upper left')
  plt.savefig(str(path)+'_acc.png')  
  plt.close()
  # summarize history for loss
  plt.plot(train_history.history['loss'])
  plt.plot(train_history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'valid'], loc='upper left')
  plt.savefig(str(path)+'_loss.png')
  plt.close()

def plot_boxplot(sp_values_per_model_NN1, sp_values_per_model_NN2, model_labels):
    # Create a box plot for Nested k-Fold CV of two models
    #  The idea is to compare this models in SP value in the test fold
    #  and then plot the two models side-by-side per fold, e.g., Fold 1 LSTM; Fold 1 BLSTM; ...,
    #  and so on.

    # Shuffle the vectors
    index_suffled_vector = []
    try:
        for i in range(len(sp_values_per_model_NN1)):
            index_suffled_vector.append(sp_values_per_model_NN1[i])
            index_suffled_vector.append(sp_values_per_model_NN2[i])
    except Exception as e:
        print("Error occurred: ", str(e))
    
    # Create a list of labels for the x-axis
    
    fold_numbers = [f'Fold {ceil((i+1)/2)} {model_labels[i%2]}' for i in range(len(index_suffled_vector))]

    # Create the boxplot
    plt.boxplot(index_shuffled_vector, labels=fold_numbers, vert=True)  # vert=True for vertical labels

    # Set labels and title
    plt.xlabel('Test Fold and Model')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Distributions for Each Test Fold and Model')

    # Rotate x-axis labels for better visibility
    plt.xticks(rotation=90)

    # Show the plot (if you want to display it immediately)
    plt.show()

def sitrep(acc_fold, loss_fold, val_fold, sp_fold, best_model, scores, sp, out):
  print('------------------------------------------------------------------------')
  print('Resultado dos folds')
  for i in range(0, len(acc_fold)):
    print('------------------------------------------------------------------------')
    print(f'> Fold {i+1} - Loss: {loss_fold[i]} - Accuracy: {acc_fold[i]}% - Validação: {val_fold[i]}% - sp: {sp_fold[best_model]}%')
  print('------------------------------------------------------------------------')
  print('Melhor modelo:')
  print(f'> Fold {best_model+1} - Loss: {loss_fold[best_model]} - Accuracy: {acc_fold[best_model]}% - Validação: {val_fold[best_model]}% - sp: {sp_fold[best_model]}%')
  print('------------------------------------------------------------------------')
  print('Média dos folds:')
  print(f'> Accuracy: {np.mean(val_fold)} (+- {np.std(val_fold)})')
  print(f'> Loss: {np.mean(loss_fold)}')
  print('------------------------------------------------------------------------')
  print('Média dos Testes:')
  print(f'> Loss: {scores[0]} - Accuracy: {scores[1] * 100}%')
  print('------------------------------------------------------------------------')
  print('indice sp: ', np.mean(sp),'%')
  print('------------------------------------------------------------------------')
  print(out)
  print('------------------------------------------------------------------------')
