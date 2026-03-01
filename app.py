import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from services.data_prep.gendata import generate_synthetic_data
from services.data_prep.data_quality import analyze_data_quality
from services.data_prep.handle_duplicates import remove_duplicates
from services.data_prep.handle_missing import apply_missing_cleaning
from services.build_model.column_selector import get_valid_targets, get_valid_predictors, detect_problem_type


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

@app.route("/prepare", methods=["GET", "POST"])
def data_preparation():
    if request.method == "POST":
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            df = pd.read_csv(filepath) if filename.endswith('.csv') else pd.read_excel(filepath)
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
                df_synthetic = pd.read_csv(output_path) if filename.endswith('.csv') else pd.read_excel(output_path)
                session['synthetic_file'] = f"synthetic_{filename}"
                session['synthetic_count'] = len(df_synthetic)
                session['data_type'] = 'Real + Synthetic'

        return redirect(url_for('data_preparation'))

    # On GET — compute preview and quality from active dataset
    uploaded_file = session.get('uploaded_file')
    active_dataset = session.get('active_dataset', 'real')
    preview_data = None
    preview_columns = None
    quality_report = None

    if uploaded_file:
        fname = session.get('synthetic_file') if active_dataset == 'synthetic' else uploaded_file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], fname) if fname else None

        if filepath and os.path.exists(filepath):
            df = pd.read_csv(filepath) if fname.endswith('.csv') else pd.read_excel(filepath)
            preview_data = df.head(5).to_dict(orient='records')
            preview_columns = df.columns.tolist()
            quality_report = analyze_data_quality(filepath)  # ← duplicate info comes from here

    return render_template("data_preparation.html", active_step=1,
                           uploaded_file=uploaded_file,
                           record_count=session.get('record_count'),
                           quality=session.get('quality'),
                           data_type=session.get('data_type'),
                           synthetic_count=session.get('synthetic_count'),
                           synthetic_file=session.get('synthetic_file'),
                           preview_data=preview_data,
                           preview_columns=preview_columns,
                           quality_report=quality_report,
                           active_dataset=active_dataset)

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

    if uploaded_file:
        fname = session.get('synthetic_file') if active_dataset == 'synthetic' else uploaded_file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath) if fname.endswith('.csv') else pd.read_excel(filepath)
            valid_targets = get_valid_targets(df)

            if target_column and target_column in df.columns:
                predictors = get_valid_predictors(df, target_column)
                problem_type = detect_problem_type(df, target_column)
                session['predictors'] = predictors['included']
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

    uploaded_file = session.get('uploaded_file')
    active_dataset = session.get('active_dataset', 'real')
    target_column = session.get('target_column')
    predictors = session.get('predictors')

    fname = session.get('synthetic_file') if active_dataset == 'synthetic' else uploaded_file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    df = pd.read_csv(filepath) if fname.endswith('.csv') else pd.read_excel(filepath)
    dataset_id = uploaded_file.rsplit('.', 1)[0]

    results = []
    for run_fn in [run_ridge, run_rf]:
        result = run_fn(
            df=df,
            target_column=target_column,
            predictor_columns=predictors,
            artifacts_dir='artifacts',
            dataset_id=dataset_id
        )
        results.append(result)

    # Mark best performing model by holdout R²
    successful = [r for r in results if r.get('success')]
    if successful:
        best = max(successful, key=lambda r: r['metrics'].get('holdout_r2', -999))
        best['recommended'] = True

    session['model_results'] = results
    session['artifact_path'] = best.get('artifact_path') if successful else None

    return redirect(url_for('build_model'))

if __name__ == "__main__":
    app.run(debug=True, port=5000)