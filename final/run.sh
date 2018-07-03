#~/bin/bash
wget -O save_finetune.zip https://www.dropbox.com/s/9dkpvttpymbxlr4/save_finetune.zip?dl=0
unzip save_finetune.zip -d save_finetune/
python3 test.py -i save_finetune/para_dict.npy -tp $1 -o $2
