from pytorch/pytorch
 
run apt update && apt install git wget unzip build-essential -y
run git clone https://github.com/bichu136/SemBERT
run mkdir 'SemBERT/snli_model_dir'

run wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1jG2RaGfLkxCPkCOkwUGgvUdBsj2kv-Ie' -O "./SemBERT/snli_model_dir/bert_config.json"
run cp "SemBERT/snli_model_dir/bert_config.json" "./SemBERT/snli_model_dir/config.json"
run wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1cFxyGwUuWrjWT8q8IrRTARiL7sSVK27q' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1cFxyGwUuWrjWT8q8IrRTARiL7sSVK27q" -O "./SemBERT/snli_model_dir/pytorch_model.bin" && rm -rf /tmp/cookies.txt 
run wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1iLP-jRsN5Eaofq01bpKyGt__pshiyNf8' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1iLP-jRsN5Eaofq01bpKyGt__pshiyNf8" -O "./srl-model-dir.zip" && rm -rf /tmp/cookies.txt
run wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=19IJ9VhNy1TxHIbzCqVF0lBwNWsOjC5xR' -O "./SemBERT/snli_model_dir/vocab.txt"
run unzip srl-model-dir.zip -d "./SemBERT"
run pip install click==7.1.1 allennlp numpy flask
run pip install allennlp_models
workdir SemBERT
cmd python app.py