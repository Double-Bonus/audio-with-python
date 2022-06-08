:: Change windows encoding for python output
chcp 1252
ECHO Started computation

python.exe .\urbanSounds.py 1 >> logs\outCnn_melNew_1.txt
timeout /t 5


python.exe .\urbanSounds.py 2 >> logs\outCnn_melNew_2.txt
timeout /t 5

python.exe .\urbanSounds.py 4 >> logs\outCnn_melNew_4.txt
timeout /t 5

python.exe .\urbanSounds.py 6 >> logs\outCnn_melNew_6.txt
timeout /t 5

python.exe .\urbanSounds.py 8 >> logs\outCnn_melNew_8.txt
timeout /t 5


ECHO Finished computation
pause


