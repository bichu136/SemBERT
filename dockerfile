from pytorch/pytorch
 
run apt update && apt install git wget zlib unzip -y
run git clone https://github.com/bichu136/SemBERT
run mkdir 'SemBERT/snli_model_dir'
run wget 'https://docs.google.com/uc?export=download&id=1jG2RaGfLkxCPkCOkwUGgvUdBsj2kv-Ie' -O "SemBERT/snli_model_dir/bert_config.json"
run cp bert_config.json config.json
run wget 'https://docs.google.com/uc?export=download&id=1cFxyGwUuWrjWT8q8IrRTARiL7sSVK27q' -O "SemBERT/snli_model_dir/pytorch_model.bin"
run wget 'https://docs.google.com/uc?export=download&id=19IJ9VhNy1TxHIbzCqVF0lBwNWsOjC5xR' -O "SemBERT/snli_model_dir/vocab.txt"
run wget 'https://docs.google.com/uc?export=download&id=1QcRntT9Eqp6FEL0Z0IYiXDc2NqWb5iHJ' -O "srl-model-dir.zip"
run unzip- unzip srl-model-dir.zip -d "SemBERT"
workdir SemBERT
cmd python app.py