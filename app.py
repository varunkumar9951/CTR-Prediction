
from flask import Flask, render_template, request
import sqlite3
import pandas as pd
import plotly.express as px
import joblib
from sklearn.linear_model import LinearRegression
import os

app = Flask(__name__)
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
DB_PATH = 'data/campaigns.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS campaigns (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        impressions INTEGER,
        clicks INTEGER,
        conversions INTEGER,
        cost REAL
    )''')
    conn.commit()
    conn.close()

def get_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM campaigns", conn)
    conn.close()
    return df

def train_model(df):
    if len(df) >= 2:
        df['ctr'] = (df['clicks'] / df['impressions']) * 100
        X = df[['impressions', 'clicks', 'cost']]
        y = df['ctr']
        model = LinearRegression()
        model.fit(X, y)
        joblib.dump(model, 'models/model.pkl')
        return True
    return False

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        data = (
            request.form['name'],
            int(request.form['impressions']),
            int(request.form['clicks']),
            int(request.form['conversions']),
            float(request.form['cost'])
        )
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO campaigns (name, impressions, clicks, conversions, cost) VALUES (?, ?, ?, ?, ?)", data)
        conn.commit()
        conn.close()

    df = get_data()
    chart = ""
    if not df.empty:
        df['CTR (%)'] = (df['clicks'] / df['impressions']) * 100
        fig = px.bar(df, x='name', y='CTR (%)', title='CTR by Campaign')
        chart = fig.to_html(full_html=False)
    return render_template("index.html", campaigns=df.to_dict(orient='records'), chart=chart)

@app.route('/predict_ctr', methods=['GET', 'POST'])
def predict_ctr():
    prediction = None
    if request.method == 'POST':
        impressions = int(request.form['impressions'])
        clicks = int(request.form['clicks'])
        cost = float(request.form['cost'])
        try:
            model = joblib.load('models/model.pkl')
            pred = model.predict([[impressions, clicks, cost]])
            prediction = round(pred[0], 2)
        except:
            prediction = "Train the model first by adding 2+ campaigns."
    return render_template("predict.html", prediction=prediction)

if __name__ == '__main__':
    init_db()
    df = get_data()
    train_model(df)
    app.run(debug=True)
