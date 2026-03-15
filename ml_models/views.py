import os
import json
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from dataset.models import Dataset
from .models import ModelResult
from . import ml_engine

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'media', 'saved_models')

MODEL_CHOICES = [
    ('cnn',         'CNN',          'fa-microchip',    'Spatial Pattern Detection'),
    ('lstm',        'LSTM',         'fa-wave-square',  'Temporal Attack Sequences'),
    ('autoencoder', 'Autoencoder',  'fa-compress-alt', 'Anomaly Detection'),
    ('hybrid',      'Hybrid',       'fa-atom',         'CNN + LSTM + AE Combined'),
]

@login_required
def train_view(request):
    datasets = Dataset.objects.filter(uploaded_by=request.user)
    if request.method == 'POST':
        dataset_id = request.POST.get('dataset_id')
        model_name = request.POST.get('model_name')
        if not dataset_id or not model_name:
            messages.error(request, 'Please select a dataset and model.')
            return render(request, 'ml_models/train.html', {'datasets': datasets, 'model_choices': MODEL_CHOICES})
        dataset = get_object_or_404(Dataset, pk=dataset_id, uploaded_by=request.user)
        try:
            result = ml_engine.train_model(model_name, dataset.file.path, request.user.id)
            mr = ModelResult.objects.create(
                user=request.user, dataset=dataset, model_name=model_name,
                accuracy=result['accuracy'], precision=result['precision'],
                recall=result['recall'], f1_score=result['f1_score'],
                confusion_matrix_img=result['confusion_matrix_img'],
                training_history=result['training_history'],
                label_classes=result['label_classes'],
            )
            messages.success(request, f'Model trained successfully! Accuracy: {result["accuracy"]}%')
            return redirect('model_result', pk=mr.pk)
        except Exception as e:
            messages.error(request, f'Training failed: {str(e)}')
    return render(request, 'ml_models/train.html', {'datasets': datasets, 'model_choices': MODEL_CHOICES})

@login_required
def model_result(request, pk):
    result = get_object_or_404(ModelResult, pk=pk, user=request.user)
    history = json.loads(result.training_history) if result.training_history else {}
    classes = json.loads(result.label_classes) if result.label_classes else []
    return render(request, 'ml_models/result.html', {'result': result, 'history': history, 'classes': classes})

@login_required
def all_results(request):
    results = ModelResult.objects.filter(user=request.user)
    return render(request, 'ml_models/all_results.html', {'results': results})

@login_required
def predict_view(request):
    saved_models = ModelResult.objects.filter(user=request.user)
    predictions = None
    classes = []
    if request.method == 'POST':
        model_id = request.POST.get('model_id')
        file = request.FILES.get('file')
        mr = get_object_or_404(ModelResult, pk=model_id, user=request.user)
        model_path = os.path.join(MODELS_DIR, f'{mr.model_name}_{request.user.id}.pkl')
        if not os.path.exists(model_path):
            messages.error(request, 'Saved model not found. Please retrain.')
            return redirect('predict')
        if not file or not file.name.endswith('.csv'):
            messages.error(request, 'Please upload a valid CSV file.')
            return redirect('predict')
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            for chunk in file.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name
        try:
            predictions, classes = ml_engine.predict_from_file(tmp_path, model_path)
            os.unlink(tmp_path)
            messages.success(request, f'Prediction complete for {len(predictions)} records.')
        except Exception as e:
            if os.path.exists(tmp_path): os.unlink(tmp_path)
            messages.error(request, f'Prediction failed: {str(e)}')
    return render(request, 'ml_models/predict.html', {'saved_models': saved_models, 'predictions': predictions, 'classes': classes, 'steps': PREDICT_STEPS})

@login_required
def compare_models(request):
    results = ModelResult.objects.filter(user=request.user)
    return render(request, 'ml_models/compare.html', {
        'results': results,
        'chart_labels': json.dumps([r.get_model_name_display() for r in results]),
        'chart_acc': json.dumps([r.accuracy for r in results]),
        'chart_prec': json.dumps([r.precision for r in results]),
        'chart_rec': json.dumps([r.recall for r in results]),
        'chart_f1': json.dumps([r.f1_score for r in results]),
    })

PREDICT_STEPS = [
    ('1', 'Upload CSV', 'Provide test data with the same feature columns as your training dataset.'),
    ('2', 'Select Model', 'Choose any of your previously trained models (CNN, LSTM, AE, or Hybrid).'),
    ('3', 'Auto Preprocess', 'The system encodes, scales, and prepares your data automatically.'),
    ('4', 'Get Results', 'Each row is classified with a label and confidence score instantly.'),
]
