:: Change windows encoding for python output
chcp 1252
ECHO Started computation

timeout /t 5
python.exe .\lstmConv2d_urbandSounds.py 1 >> logs\outLstmCon_stft_1.txt

timeout /t 5
python.exe .\lstmConv2d_urbandSounds.py 2 >> logs\outLstmCon_stft_2.txt

timeout /t 5
python.exe .\lstmConv2d_urbandSounds.py 3 >> logs\outLstmCon_stft_3.txt

timeout /t 5
python.exe .\lstmConv2d_urbandSounds.py 4 >> logs\outLstmCon_stft_4.txt

timeout /t 5
python.exe .\lstmConv2d_urbandSounds.py 5 >> logs\outLstmCon_stft_5.txt

timeout /t 5
python.exe .\lstmConv2d_urbandSounds.py 6 >> logs\outLstmCon_stft_6.txt

timeout /t 5
python.exe .\lstmConv2d_urbandSounds.py 7 >> logs\outLstmCon_stft_7.txt

timeout /t 5
python.exe .\lstmConv2d_urbandSounds.py 8 >> logs\outLstmCon_stft_8.txt

timeout /t 5
python.exe .\lstmConv2d_urbandSounds.py 9 >> logs\outLstmCon_stft_9.txt

timeout /t 5
python.exe .\lstmConv2d_urbandSounds.py 10 >> logs\outLstmCon_stft_10.txt


ECHO Finished computation
pause

timeout /t 5

