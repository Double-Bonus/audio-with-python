:: Change windows encoding for python output
chcp 1252
ECHO Started computation

python.exe .\lstm_ann_urbSound.py 5 >> logs\outCNN_raw_6.txt

timeout /t 5
python.exe .\lstm_ann_urbSound.py 6 >> logs\outCNN_raw_7.txt

timeout /t 5
python.exe .\lstm_ann_urbSound.py 7 >> logs\outCNN_raw_8.txt

timeout /t 5
python.exe .\lstm_ann_urbSound.py 8 >> logs\outCNN_raw_9.txt

timeout /t 5
python.exe .\lstm_ann_urbSound.py 9 >> logs\outCNN_raw_10.txt

ECHO Finished computation
pause

timeout /t 5

