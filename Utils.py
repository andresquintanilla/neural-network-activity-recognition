# %%
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 20:47:23 2019

@author: AndresQuintanilla
"""
# Importing necesary libraries/modules
import os
import tarfile
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix

def make_tardir(path, tar_name):
    with tarfile.open(tar_name, "w:gz") as tar_handle:
        for root, dirs, files in os.walk(path):
            for file in files:
                tar_handle.add(os.path.join(root, file))
        tar_handle.close()

"""
This function plots a 3D graph with the PCA (principal components)
"""
def plot_3d_pca(df, activity_types, plot_angle=310, figsize_i=(15,15)):
    fig = plt.figure(figsize = figsize_i)
    ax = fig.add_subplot(1,1,1, projection='3d') 
    ax.set_xlabel('PC 1', fontsize = 15)
    ax.set_ylabel('PC 2', fontsize = 15)
    ax.set_zlabel('PC 3', fontsize = 15)
    ax.set_title('Principal Components', fontsize = 20)
    for actId in df['ActivityID'].unique():
        idx = df['ActivityID'] == actId
        ax.scatter(df.loc[idx, 'PCA 1'], df.loc[idx, 'PCA 2'], df.loc[idx, 'PCA 3'], s = 50)
    ax.legend(list(np.array(activity_types)[df['ActivityID'].unique()]))
    ax.view_init(30, plot_angle)
    #plt.show()
    
    return plt


"""
This function plots Activity samples broken into frames of specific range.
"""
def plot_activity_frames(df, variables, activityList, activity_types, samplesPerFrame, totalFrames, figsize_i=(20,25)):
    plt.figure(figsize=figsize_i)
    for idx, activity in enumerate(activityList):
        df_temp = df[df['ActivityID'] == activity][df.columns.difference(['ActivityID'])]
        temp_data = []
        for x in range(int(df_temp.shape[0]/samplesPerFrame)):
            temp_data.append(np.array(df_temp.iloc[(samplesPerFrame*x):(samplesPerFrame*x+samplesPerFrame)][variables]))
        temp_data = np.array(temp_data)
        d0,d1,d2 = temp_data.shape
        temp_data = temp_data.reshape(d0,d1,d2,1)  
        
        for i in range(1,totalFrames+1):
            ax = plt.subplot(len(df['ActivityID'].unique()), totalFrames, totalFrames*idx+i)
            ax.plot(temp_data[i].reshape(d1,d2))
            if i != 1:
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.set_xticks([])
            if i == int(totalFrames/2):
                ax.set_title(activity_types[activity])
    #plt.show()
    
    return plt

"""
This function brakes samples in a dataset into 'chunks' of window frames. 
It returns to Numpy Arrays one for raw data (shape={N,samplesPerFrame,6}) and one for label data (shape={N,1})
"""
def extract_sample_frames(df, samplesPerFrame):
    temp_data, temp_label = [], []
    
    for act in df['ActivityID'].unique():
        df_temp = df[df['ActivityID'] == act]
   
        for x in range(int(df_temp.shape[0]/samplesPerFrame)):
            temp_data.append(np.array(df_temp.iloc[(samplesPerFrame*x):(samplesPerFrame*x+samplesPerFrame)][df.columns.difference(['ActivityID'])]))
            temp_label.append(int(np.median(np.array(df_temp.iloc[(samplesPerFrame*x):(samplesPerFrame*x+samplesPerFrame)]['ActivityID']))))
   
    temp_data = np.array(temp_data)
    temp_label = np.array(temp_label)
    
    d0,d1,d2 = temp_data.shape
    temp_data = temp_data.reshape(d0,d1,d2,1)

    return temp_data, temp_label

"""
This function plots the accuracy and loss history of a Convolutional Neural Network model.
"""
def plot_network_history(model_history,figsize_i=(15,5)):
    fig, axs = plt.subplots(1,2,figsize=figsize_i)
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])
    axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy (%d%%)' % (100*max(model_history.history['val_accuracy'])),fontsize=20)
    axs[0].set_ylabel('Accuracy',fontsize=20)
    axs[0].set_xlabel('Epoch',fontsize=20)
    axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1),len(model_history.history['accuracy'])/10)
    axs[0].legend(['train', 'val'], loc='best', fontsize=15)
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss (%.2f)' % (min(model_history.history['val_loss'])),fontsize=20)
    axs[1].set_ylabel('Loss',fontsize=20)
    axs[1].set_xlabel('Epoch',fontsize=20)
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best', fontsize=15)
    
    fig.tight_layout()
    
    #plt.show()
    
    return plt


"""
This function prints and plots the confusion matrix.
Normalization can be applied by setting `normalize=True`.
"""
def plot_confusion_matrix(y_true, y_pred, classes, cmap=plt.cm.Blues, normalize=False, figsize_i=(10,10)):
    # Plot the confusion matrix based on the test data and predicted values
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=figsize_i)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    #ax.figure.colorbar(im, ax=ax)
    
    # We want to show all ticks...
    plt.title("Confusion Matrix", fontsize=25)
    plt.ylabel("True label", fontsize=25)
    plt.xlabel("Predicted label",fontsize=25)
    ax.set(xticks=np.arange(cm.shape[1]),yticks=np.arange(cm.shape[0]),xticklabels=classes, yticklabels=classes)
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",fontsize=20)
    fig.tight_layout()

    #plt.show()
    
    return plt