:: Change windows encoding for python output
chcp 1252
ECHO Started computation

python.exe .\urbanSounds.py >> logs\outCnn.txt

ECHO Finished computation