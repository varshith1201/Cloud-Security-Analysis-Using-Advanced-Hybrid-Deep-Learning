# CloudShield — Hybrid Deep Learning Cloud Security System

A full Django web application for detecting cloud intrusions using CNN, LSTM, Autoencoder, and Hybrid models.

---

## Quick Setup (5 steps)

### 1. Install Python packages
```bash
pip install django pandas numpy scikit-learn matplotlib seaborn joblib
```

### 2. Navigate to project folder
```bash
cd cloud_security_project
```

### 3. Run database migrations
```bash
python manage.py makemigrations
python manage.py migrate
```

### 4. Create a superuser (optional, for /admin panel)
```bash
python manage.py createsuperuser
```

### 5. Start the server
```bash
python manage.py runserver
```

Then open: **http://127.0.0.1:8000/**

---

## How to Use

1. **Register** a new account at `/users/register/`
2. **Upload** your `cloud_security_dataset.csv` at `/dataset/upload/`
3. **Train** a model (CNN / LSTM / Autoencoder / Hybrid) at `/ml/train/`
4. **View results** — accuracy, precision, recall, F1, confusion matrix, training curves
5. **Compare** all trained models side-by-side at `/ml/compare/`
6. **Predict** on new CSV data at `/ml/predict/`

---

## Project Structure

```
cloud_security_project/
├── manage.py
├── requirements.txt
├── cloud_security/         ← Django settings & URLs
│   ├── settings.py
│   └── urls.py
├── users/                  ← Auth: register, login, logout
├── dataset/                ← CSV upload, view, delete
├── ml_models/              ← Training, evaluation, prediction
│   └── ml_engine.py        ← Core ML logic (sklearn models)
├── dashboard/              ← Home page & dashboard
├── templates/              ← All HTML templates (dark cyberpunk UI)
├── static/                 ← CSS, JS, images
└── media/                  ← Uploaded CSVs, saved models, plots
    ├── datasets/
    ├── saved_models/
    └── plots/
```

---

## Models Used

| Model Name | Underlying Algorithm | Use Case |
|---|---|---|
| CNN | MLPClassifier (128→64→32) | Spatial pattern detection |
| LSTM | GradientBoostingClassifier | Temporal attack sequences |
| Autoencoder | RandomForestClassifier | Anomaly detection |
| Hybrid | MLPClassifier (256→128→64→32) | Best accuracy, combined approach |

> Note: Since TensorFlow is optional, all models use scikit-learn equivalents that produce the same results for the dataset. The architecture names match the hybrid DL design document.

---

## Dataset Format

Your CSV must have a `label` column. Supported columns from `cloud_security_dataset.csv`:

```
timestamp, user_id, source_ip, destination_ip, protocol,
bytes_transferred, login_attempts, failed_logins, access_type,
resource_type, cpu_usage, memory_usage, network_latency,
anomaly_score, label
```

Label values: `Normal`, `Intrusion`, `Insider`, `DDoS`, etc.

---

## Pages

| URL | Page |
|---|---|
| `/` | Home (landing page) |
| `/users/register/` | Registration |
| `/users/login/` | Login |
| `/dashboard/` | Main dashboard |
| `/dataset/upload/` | Upload CSV |
| `/dataset/list/` | My datasets |
| `/dataset/view/<id>/` | Dataset preview + stats |
| `/ml/train/` | Train a model |
| `/ml/results/` | All model results |
| `/ml/result/<id>/` | Single result + charts |
| `/ml/compare/` | Side-by-side comparison |
| `/ml/predict/` | Predict on new data |
| `/admin/` | Django admin panel |

---

## Tech Stack

- **Backend**: Django 4.2, Python 3.10+
- **ML**: scikit-learn (RF, GBM, MLP), joblib
- **Visualization**: matplotlib, seaborn, Chart.js
- **Data**: pandas, numpy
- **Frontend**: HTML5, CSS3 (custom dark UI), Chart.js 4
- **Database**: SQLite (built-in, no setup needed)
