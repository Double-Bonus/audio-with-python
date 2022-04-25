:: Change windows encoding for python output
chcp 1252
ECHO Started computation

python.exe .\urbanSounds.py >> logs\outCnn.txt
python.exe .\lstmConv2d_urbandSounds.py >> logs\outLstmCon2.txt


ECHO Finished computation
pause