import os
import flask as flk
import json
import ast

app = flk.Flask(__name__)
@app.route('/')
def GUI():
    x = [i for i in range(10)]
    return flk.render_template('index.html',suggest_sentences=x)
@app.route('/acceptpost',methods=["POST"])
def test():
    data = flk.request.get_data()
    dict_str = data.decode("UTF-8")
    mydata = ast.literal_eval(dict_str)
    sent1 = mydata["param1"]
    sent2 = mydata["param2"]
    d = {"result":sent1+sent2}
    return flk.jsonify(d)

if __name__ == '__main__':
    app.run(debug=True,port=int(os.environ.get('PORT', 13600)),host='0.0.0.0')