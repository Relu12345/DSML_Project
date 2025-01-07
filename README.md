
# Chaos Theory in Chess

Acest proiect analizeaza Chaos Theory aplicata in jocul de sah.

## Cerințe preliminare

- Stockfish
- Dataset cu partide de sah

## Instalare și Configurare

1. Clonati repository-ul:

git clone https://github.com/Relu12345/DSML_Project.git
cd DSML_Project


2. Instalati dependintele Python:

pip install -r requirements.txt


3. Configurati dataset-ul:
   - Creati un folder `Dataset` în directorul root al proiectului
   - Descarcati fisierul de dataset din link-ul: https://database.lichess.org/standard/lichess_db_standard_rated_2017-03.pgn.zst
   - Dezarhivati fisierul, ruland `decompress_dataset.py`

4. Instalati Stockfish:
   - Descarcati binaries Stockfish de aici: https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-windows-x86-64-avx2.zip
   - Extrageti arhiva in directorul root al proiectului
   - Asigurati-va ca aveti un folder `stockfish` in directorul root
