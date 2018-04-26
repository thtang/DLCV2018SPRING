wget -O fcn8s_25ep_model.pkt https://www.dropbox.com/s/bnaqrblcpu42aqq/fcn8s_25ep_model.pkt?dl=0
python3 fcn32_inference.py 0 ./fcn8s_25ep_model.pkt $1 $2