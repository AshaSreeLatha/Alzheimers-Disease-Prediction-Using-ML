from ast import alias
from concurrent.futures import process
from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, HttpResponse
from django.contrib import messages

import Alzheimers_Disease_Prediction

from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings
import os


def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(
                loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})

def DatasetView(request):
    path = settings.MEDIA_ROOT + "//" + 'alzheimers_prediction_dataset.csv'
    df = pd.read_csv(path)
    df = df.to_html
    return render(request, 'users/viewdataset.html', {'data': df})

import pandas as pd
df = pd.read_csv(r'media\alzheimers_prediction_dataset.csv')
x = df.drop('Alzheimer_Diagnosis', axis=1)
y=df['Alzheimer_Diagnosis']

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report , confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss, roc_curve, auc
from sklearn.linear_model import LogisticRegression
import numpy as np
import seaborn as sns

path = r'balanced_data.csv'
df = pd.read_csv(path)
df.fillna(0, inplace=True)

    
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
model = LogisticRegression(random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
# Create your views here.
from django.shortcuts import render
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


def Training(request):
    # Define model at the start of the function
    model = LogisticRegression(max_iter=1000, random_state=42)

    # Train/test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Fit the model on the training data
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.2f}")      
    # Make predictions on the test data
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Confusion matrix plot
    # plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=['No Alzheimer\'s', 'Alzheimer\'s'], 
                yticklabels=['No Alzheimer\'s', 'Alzheimer\'s'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # Train/test accuracies
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"Train Accuracy: {train_acc * 100:.2f}%")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    # Cross-validation
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for train_index, val_index in kf.split(x, y):
        x_train_fold, x_val_fold = x.iloc[train_index], x.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
        
        # Train the model on each fold
        model.fit(x_train_fold, y_train_fold)
        
        # Train and validation metrics
        y_train_pred_fold = model.predict(x_train_fold)
        y_train_prob_fold = model.predict_proba(x_train_fold)
        train_accuracies.append(accuracy_score(y_train_fold, y_train_pred_fold))
        train_losses.append(log_loss(y_train_fold, y_train_prob_fold))

        y_val_pred_fold = model.predict(x_val_fold)
        y_val_prob_fold = model.predict_proba(x_val_fold)
        val_accuracies.append(accuracy_score(y_val_fold, y_val_pred_fold))
        val_losses.append(log_loss(y_val_fold, y_val_prob_fold))

    # Plot Training vs Validation Accuracy
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy', marker='o')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training vs Validation Accuracy')

    # Plot Training vs Validation Loss
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Fold')
    plt.ylabel('Log Loss')
    plt.legend()
    plt.title('Training vs Validation Loss')
    plt.tight_layout()
    plt.show()

    # ROC curve and AUC
    y_test_prob = model.predict_proba(x_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    # Return context to template
    context = {
        'accuracy':accuracy,
        'train_acc': train_acc,
        'test_acc': test_acc
    }
    return render(request, 'users/training.html', context)


def Prediction(request):
    if request.method == "POST":
        ag = request.POST.get('age')
        cl = request.POST.get('cholesterol_levels')
        cd = request.POST.get('chest_discomfort')
        ohf1 = request.POST.get('other_health_factor_1')
        ohf2 = request.POST.get('other_health_factor_2')
        # ag = int(ag) if ag else 0
        # cl = int(cl) if cl else 0
        cd = 1 if cd == 'Yes' else 0
        # ohf1 = float(ohf1) if ohf1 else 0
        # ohf2 = float(ohf2) if ohf2 else 0
        
        input_df = pd.DataFrame({
            'Age': [ag],
            'Cholesterol_Levels': [cl],
            'Chest_Discomfort': [cd],
            'Other_Health_Factor_1': [ohf1],
            'Other_Health_Factor_2': [ohf2]
        })
        print(input_df)
        print("Input DataFrame before encoding:", input_df)
        op = model.predict(input_df)
        prediction_label = "Patient contain Alzheimer's" if int(op[0]) == 1 else "Patient does not contain Alzheimer's"
        print("Prediction label:", prediction_label)
        context = {
        'prediction': prediction_label
        }

        return render(request, 'users/predict_form.html', context)

    return render(request, 'users/predict_form.html')

