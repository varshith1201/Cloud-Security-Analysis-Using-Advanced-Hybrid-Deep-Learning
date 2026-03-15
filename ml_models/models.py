from django.db import models
from django.contrib.auth.models import User
from dataset.models import Dataset

class ModelResult(models.Model):
    MODEL_CHOICES = [
        ('cnn', 'CNN'),
        ('lstm', 'LSTM'),
        ('autoencoder', 'Autoencoder'),
        ('hybrid', 'Hybrid (CNN+LSTM+AE)'),
    ]
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    model_name = models.CharField(max_length=50, choices=MODEL_CHOICES)
    accuracy = models.FloatField(default=0)
    precision = models.FloatField(default=0)
    recall = models.FloatField(default=0)
    f1_score = models.FloatField(default=0)
    confusion_matrix_img = models.CharField(max_length=500, blank=True)
    training_history = models.TextField(blank=True)  # JSON
    label_classes = models.TextField(blank=True)  # JSON
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.model_name} - {self.accuracy:.2f}% ({self.created_at.date()})"

    class Meta:
        ordering = ['-created_at']
