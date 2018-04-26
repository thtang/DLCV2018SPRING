wget -O fcn32s_25ep_model.pkt https://www.dropbox.com/s/171qljyl5jnpzd9/fcn32s_25ep_model.pkt?dl=0
python3 fcn32_inference.py 0 ./fcn32s_25ep_model.pkt $1 $2