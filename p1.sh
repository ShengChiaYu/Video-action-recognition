# TODO: create shell script for Problem 1
wget -O p1_best.pth 'https://www.dropbox.com/s/vn1y35qo8dnu5pr/p1_best.pth?dl=1'
python3 predict.py p1 $1 $2 $3
