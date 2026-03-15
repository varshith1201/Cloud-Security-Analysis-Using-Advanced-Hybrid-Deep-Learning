import json
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from dataset.models import Dataset
from ml_models.models import ModelResult

def home_view(request):
    return render(request, 'dashboard/home.html')

@login_required
def dashboard_view(request):
    datasets = Dataset.objects.filter(uploaded_by=request.user)
    results = ModelResult.objects.filter(user=request.user)
    latest_results = results[:5]

    # Chart data for model comparison
    model_names = [r.get_model_name_display() for r in results]
    accuracies = [r.accuracy for r in results]

    # Label distribution from latest dataset
    label_dist = {}
    if datasets.exists():
        import pandas as pd
        try:
            df = pd.read_csv(datasets.first().file.path)
            if 'label' in df.columns:
                label_dist = df['label'].value_counts().to_dict()
        except Exception:
            pass

    context = {
        'total_datasets': datasets.count(),
        'total_models': results.count(),
        'best_accuracy': max([r.accuracy for r in results], default=0),
        'latest_results': latest_results,
        'chart_labels': json.dumps(model_names),
        'chart_acc': json.dumps(accuracies),
        'label_dist_keys': json.dumps(list(label_dist.keys())),
        'label_dist_vals': json.dumps(list(label_dist.values())),
    }
    return render(request, 'dashboard/dashboard.html', context)
