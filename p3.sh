# TODO: create shell script for Problem 3
wget -O p3_best.pth 'https://www.dropbox.com/s/osoaau1ob7ucwvy/p3_best.pth?dl=1'
python3 predict.py p3 $1 $2
