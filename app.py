from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("model/fraudmodel.joblib")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/submit", methods=["POST"])
def predict():
    try:
        # data gatherinh
        transaction_price = float(request.form.get("transaction_price", 0))
        time_of_day = int(request.form.get("time_of_day", 0))
        foreign_transaction = int(request.form.get("foreign_transaction", 0))
        online_purchase = int(request.form.get("online_purchase", 0))
    except:
        print("Something happened")
        return render_template("index.html", prediction="An Error Occurred")
        
    # data frame with the user inputs
    input_data = pd.DataFrame([{
        "transaction_price": transaction_price,
        "time_of_day": time_of_day,
        "foreign_transaction": foreign_transaction,
        "online_purchase": online_purchase
    }])


    # the ai predictingh
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        result = "The transaction was likely fraud"
    elif prediction == 0:
        result = "The transaction was likely not fraud"
    else:
        result = "An Error Occurred"


    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)