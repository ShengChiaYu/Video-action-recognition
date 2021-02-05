# TODO: create shell script for Problem 2
wget -O p2_best_4.pth 'https://www.dropbox.com/s/1c4e6usa6o40648/p2_best_4.pth?dl=1'
python3 predict.py p2 $1 $2 $3
