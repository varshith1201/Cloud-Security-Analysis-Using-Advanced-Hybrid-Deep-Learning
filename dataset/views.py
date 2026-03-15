import os
import json
import pandas as pd
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from .models import Dataset

@login_required
def upload_dataset(request):
    if request.method == 'POST':
        file = request.FILES.get('file')
        name = request.POST.get('name', file.name if file else 'Dataset')
        description = request.POST.get('description', '')

        if not file:
            messages.error(request, 'Please select a file.')
            return redirect('upload_dataset')

        if not file.name.endswith('.csv'):
            messages.error(request, 'Only CSV files are allowed.')
            return redirect('upload_dataset')

        dataset = Dataset.objects.create(
            name=name,
            file=file,
            uploaded_by=request.user,
            description=description
        )

        try:
            df = pd.read_csv(dataset.file.path)
            dataset.rows = len(df)
            dataset.columns = len(df.columns)
            dataset.save()
        except Exception as e:
            pass

        messages.success(request, f'Dataset "{name}" uploaded successfully!')
        return redirect('view_dataset', pk=dataset.pk)

    return render(request, 'dataset/upload.html')

@login_required
def list_datasets(request):
    datasets = Dataset.objects.filter(uploaded_by=request.user)
    return render(request, 'dataset/list.html', {'datasets': datasets})

@login_required
def view_dataset(request, pk):
    dataset = get_object_or_404(Dataset, pk=pk, uploaded_by=request.user)
    try:
        df = pd.read_csv(dataset.file.path)
        columns = list(df.columns)
        preview = df.head(20).to_dict('records')
        dtypes = df.dtypes.astype(str).to_dict()
        nulls = df.isnull().sum().to_dict()
        label_dist = {}
        if 'label' in df.columns:
            label_dist = df['label'].value_counts().to_dict()
        stats = {
            'shape': df.shape,
            'columns': columns,
            'dtypes': dtypes,
            'nulls': nulls,
            'label_dist': label_dist,
        }
    except Exception as e:
        preview = []
        stats = {}
        columns = []
        messages.error(request, f'Error reading dataset: {str(e)}')

    context = {
        'dataset': dataset,
        'preview': preview,
        'stats': stats,
        'columns': columns,
    }
    return render(request, 'dataset/view.html', context)

@login_required
def delete_dataset(request, pk):
    dataset = get_object_or_404(Dataset, pk=pk, uploaded_by=request.user)
    if request.method == 'POST':
        name = dataset.name
        if os.path.exists(dataset.file.path):
            os.remove(dataset.file.path)
        dataset.delete()
        messages.success(request, f'Dataset "{name}" deleted.')
        return redirect('list_datasets')
    return render(request, 'dataset/confirm_delete.html', {'dataset': dataset})
