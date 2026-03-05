import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from services.data_prep.gendata import generate_synthetic_data
from services.data_prep.handle_duplicates import remove_duplicates
from services.data_prep.handle_missing import apply_missing_cleaning
from services.data_prep.data_quality import analyze_data_quality
from services.build_model.column_selector import get_valid_targets, get_valid_predictors, detect_problem_type, compute_feature_importance
from services.data_prep.handle_column import format_preview_records


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.secret_key = "simucast-dev-key"

ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    session.clear()
    return render_template("landing.html")

@app.route("/change-dataset")
def change_dataset():
    session.clear()
    return redirect(url_for('data_preparation'))

def load_dataframe(filepath):
    if filepath.endswith('.csv'):
        peek = pd.read_csv(filepath, header=None, nrows=3)
        row0 = peek.iloc[0].astype(str).tolist()
        row1 = peek.iloc[1].astype(str).tolist()

        empty_in_row0 = sum(1 for v in row0 if str(v).strip() in ('', 'nan'))

        # Also catch merged headers where row0 has fewer unique values than row1
        # (group headers repeat across columns, real headers are all distinct)
        unique_row0 = len(set(v.strip() for v in row0 if v.strip() not in ('', 'nan')))
        unique_row1 = len(set(v.strip() for v in row1 if v.strip() not in ('', 'nan')))

        if empty_in_row0 > 0 or unique_row0 < unique_row1:
            df = pd.read_csv(filepath, header=1)
        else:
            df = pd.read_csv(filepath, header=0)
        return df

    # Excel — same logic
    peek = pd.read_excel(filepath, header=None, nrows=3)
    row0 = peek.iloc[0].astype(str).tolist()
    row1 = peek.iloc[1].astype(str).tolist()

    empty_in_row0 = sum(1 for v in row0 if str(v).strip() in ('', 'nan'))
    unique_row0 = len(set(v.strip() for v in row0 if v.strip() not in ('', 'nan')))
    unique_row1 = len(set(v.strip() for v in row1 if v.strip() not in ('', 'nan')))

    if empty_in_row0 > 0 or unique_row0 < unique_row1:
        df = pd.read_excel(filepath, header=1)
    else:
        df = pd.read_excel(filepath, header=0)
    return df

@app.route("/prepare", methods=["GET", "POST"])
def data_preparation():
    if request.method == "POST":
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            df = load_dataframe(filepath)  # ← use load_dataframe
            quality = round((df.notna().sum().sum() / df.size) * 100)

            session['uploaded_file'] = filename
            session['record_count'] = len(df)
            session['quality'] = quality
            session['data_type'] = 'Real'
            session['active_dataset'] = 'real'

            use_synthetic = request.form.get('use_synthetic')
            if use_synthetic:
                num_rows = int(request.form.get('synthetic_count', 500))
                output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"synthetic_{filename}")
                generate_synthetic_data(input_path=filepath, num_rows=num_rows, output_path=output_path)
                df_synthetic = load_dataframe(output_path)  # ← use load_dataframe
                session['synthetic_file'] = f"synthetic_{filename}"
                session['synthetic_count'] = len(df_synthetic)
                session['data_type'] = 'Real + Synthetic'

        return redirect(url_for('data_preparation'))

    # On GET
    uploaded_file = session.get('uploaded_file')
    active_dataset = session.get('active_dataset', 'real')
    quality_report = None  # ← add this
    preview_data = None
    preview_columns = None
    

    if uploaded_file:
        fname = session.get('synthetic_file') if active_dataset == 'synthetic' else uploaded_file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], fname) if fname else None

        if filepath and os.path.exists(filepath):
            df = load_dataframe(filepath)
            preview_columns = df.columns.tolist()
            preview_data = format_preview_records(df, n=5)
            quality_report = analyze_data_quality(df)

    return render_template("data_preparation.html", active_step=1,
                           uploaded_file=uploaded_file,
                           record_count=session.get('record_count'),
                           quality=session.get('quality'),
                           data_type=session.get('data_type'),
                           synthetic_count=session.get('synthetic_count'),
                           synthetic_file=session.get('synthetic_file'),
                           active_dataset=active_dataset,
                           quality_report=quality_report,
                           preview_data=preview_data,
                           preview_columns=preview_columns)

@app.route("/select-dataset", methods=["POST"])
def select_dataset():
    session['active_dataset'] = request.form.get('active_dataset', 'real')
    return redirect(url_for('data_preparation'))

@app.route("/clean/duplicates", methods=["POST"])
def clean_duplicates():
    active_dataset = session.get('active_dataset', 'real')
    fname = session.get('synthetic_file') if active_dataset == 'synthetic' else session.get('uploaded_file')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], fname)

    result = remove_duplicates(filepath)
    session['clean_message'] = f"Removed {result['removed_count']} duplicate rows. {result['remaining_count']} rows remaining."

    if active_dataset == 'synthetic':
        session['synthetic_count'] = result['remaining_count']
    else:
        session['record_count'] = result['remaining_count']

    return redirect(url_for('data_preparation'))



@app.route("/clean/missing", methods=["POST"])
def clean_missing():
    active_dataset = session.get('active_dataset', 'real')
    fname = session.get('synthetic_file') if active_dataset == 'synthetic' else session.get('uploaded_file')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], fname)

    options = {
        'remove_rows': request.form.get('remove_rows'),
        'mean':        request.form.get('mean_imputation'),
        'median':      request.form.get('median_imputation'),
        'mode':        request.form.get('mode_imputation'),
    }

    result = apply_missing_cleaning(filepath, options)

    if active_dataset == 'synthetic':
        session['synthetic_count'] = result['remaining_rows']
    else:
        session['record_count'] = result['remaining_rows']

    session['clean_message'] = f"Fixed {result['fixed']} missing values. {result['removed_rows']} rows removed."
    return redirect(url_for('data_preparation'))

@app.route("/build-model/set-target", methods=["POST"])
def set_target():
    session['target_column'] = request.form.get('target_column')
    session.pop('target_confirmed', None)
    session.pop('model_results', None)
    return redirect(url_for('build_model'))




@app.route("/build-model")
def build_model():
    uploaded_file = session.get('uploaded_file')
    active_dataset = session.get('active_dataset', 'real')
    target_column = session.get('target_column')
    valid_targets = []
    predictors = None
    problem_type = None
    importance = None

    if uploaded_file:
        fname = session.get('synthetic_file') if active_dataset == 'synthetic' else uploaded_file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        if os.path.exists(filepath):
            df = load_dataframe(filepath)  # ← was pd.read_csv/read_excel
            valid_targets = get_valid_targets(df)

            if target_column and target_column in df.columns:
                predictors    = get_valid_predictors(df, target_column)
                problem_type  = detect_problem_type(df, target_column)
                importance    = compute_feature_importance(df, target_column)   # ← new
                session['predictors']   = predictors['included']
                session['problem_type'] = problem_type

    return render_template("build_model.html", active_step=2,
                           uploaded_file=uploaded_file,
                           active_dataset=active_dataset,
                           record_count=session.get('record_count'),
                           synthetic_count=session.get('synthetic_count'),
                           valid_targets=valid_targets,
                           target_column=target_column,
                           predictors=predictors,
                           problem_type=problem_type,
                           importance=importance,
                           target_confirmed=session.get('target_confirmed'),
                           model_results=session.get('model_results'))

@app.route("/build-model/reset", methods=["POST"])
def reset_model():
    session.pop('target_confirmed', None)
    session.pop('target_column', None)
    session.pop('model_results', None)
    session.pop('predictors', None)
    session.pop('problem_type', None)
    return redirect(url_for('build_model'))


from services.build_model.model_list.model1 import run as run_ridge
from services.build_model.model_list.model2 import run as run_rf

@app.route("/build-model/confirm", methods=["POST"])
def confirm_model():
    session['target_confirmed'] = True

    selected_predictors = request.form.getlist('predictors')
    if selected_predictors:
        session['predictors'] = selected_predictors

    uploaded_file = session.get('uploaded_file')
    active_dataset = session.get('active_dataset', 'real')
    target_column = session.get('target_column')
    predictors = session.get('predictors')

    fname = session.get('synthetic_file') if active_dataset == 'synthetic' else uploaded_file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    df = load_dataframe(filepath)  # ← fix
    dataset_id = uploaded_file.rsplit('.', 1)[0]

    results = []
    for run_fn in [run_ridge, run_rf]:
        result = run_fn(df=df, target_column=target_column,
                        predictor_columns=predictors,
                        artifacts_dir='artifacts', dataset_id=dataset_id)
        results.append(result)
        
    selected_predictors = request.form.getlist('predictors')
    

    successful = [r for r in results if r.get('success')]
    if successful:
        best = max(successful, key=lambda r: r['metrics'].get('holdout_r2', -999))
        best['recommended'] = True

    session['model_results'] = results
    session['artifact_path'] = best.get('artifact_path') if successful else None
    session['selected_model_key'] = best.get('model_key') if successful else None

    session['model_results'] = results
    session['artifact_path'] = best.get('artifact_path') if successful else None
    return redirect(url_for('build_model'))


@app.route("/build-model/retrain", methods=["POST"])
def retrain_model():
    uploaded_file = session.get('uploaded_file')
    active_dataset = session.get('active_dataset', 'real')
    target_column = session.get('target_column')
    predictors = session.get('predictors')

    fname = session.get('synthetic_file') if active_dataset == 'synthetic' else uploaded_file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    df = load_dataframe(filepath)  # ← fix
    dataset_id = uploaded_file.rsplit('.', 1)[0]

    train_pct = int(request.form.get('test_size', 80))
    test_size = (100 - train_pct) / 100
    ridge_alpha = float(10 ** float(request.form.get('ridge_alpha', 0)))
    n_estimators = int(request.form.get('n_estimators', 300))
    max_depth_raw = request.form.get('max_depth', 'None')
    max_depth = None if max_depth_raw == 'None' else int(max_depth_raw)
    min_samples_leaf = int(request.form.get('min_samples_leaf', 1))

    result_ridge = run_ridge(df=df, target_column=target_column,
                             predictor_columns=predictors,
                             artifacts_dir='artifacts', dataset_id=dataset_id,
                             test_size=test_size, ridge_alpha=ridge_alpha)

    result_rf = run_rf(df=df, target_column=target_column,
                       predictor_columns=predictors,
                       artifacts_dir='artifacts', dataset_id=dataset_id,
                       test_size=test_size, n_estimators=n_estimators,
                       max_depth=max_depth, min_samples_leaf=min_samples_leaf)

    results = [result_ridge, result_rf]
    successful = [r for r in results if r.get('success')]
    if successful:
        best = max(successful, key=lambda r: r['metrics'].get('holdout_r2', -999))
        best['recommended'] = True

    session['model_results'] = results
    session['artifact_path'] = best.get('artifact_path') if successful else None
    session['selected_model_key'] = best.get('model_key') if successful else None

    session['model_results'] = results
    return redirect(url_for('build_model'))


@app.route("/build-model/select-model", methods=["POST"])
def select_model():
    session['selected_model_key'] = request.form.get('model_key')
    session['artifact_path'] = request.form.get('artifact_path')
    return redirect(url_for('build_model'))



if __name__ == "__main__":
    app.run(debug=True, port=5000)