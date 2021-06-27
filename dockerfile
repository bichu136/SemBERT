from pytorch/pytorch
 
run apt update && apt install git wget -y

run wget 'https://docs.google.com/uc?export=download&id=1jG2RaGfLkxCPkCOkwUGgvUdBsj2kv-Ie' -O bert_config.json
run cp bert_config.json config.json
run wget 'https://docs.google.com/uc?export=download&id=1cFxyGwUuWrjWT8q8IrRTARiL7sSVK27q' -O pytorch_model.bin
run wget 'https://docs.google.com/uc?export=download&id=19IJ9VhNy1TxHIbzCqVF0lBwNWsOjC5xR' -O vocab.txt
run wget 'https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz' -O srl-model-dir.tar.gz
run git clone 