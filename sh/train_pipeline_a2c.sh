python models/utils/sinus_generator.py --amplitude 50 --period 0.2 --offset 900 --noise 0 --nb_sin 1
python models/utils/sinus_generator.py --amplitude 150 --period 0.4 --offset 1000 --noise 0 --nb_sin 1
python models/utils/sinus_generator.py --amplitude 50 --period 0.2 --offset 900 --noise 0.05 --nb_sin 1
python models/utils/sinus_generator.py --amplitude 50 --period 0.2 --offset 900 --noise 0 --nb_sin 2
python models/utils/sinus_generator.py --amplitude 50 --period 0.2 --offset 900 --noise 0.08 --nb_sin 1

python main.py --config config_a2c --df_name generated_data/sinus_l0.2_M50_h900_noise0_nbsin1.csv --load 0 --save 1 --num_episode 10000 --save_path models_records/A2Csinus0 --lr 1
python main.py --config config_a2c --df_name generated_data/sinus_l0.4_M150_h1000_noise0_nbsin1.csv --load 1 --save 1 --num_episode 10000 --save_path models_records/A2Csinus1 --load_path models_records/A2Csinus0  --lr 1
python main.py --config config_a2c --df_name generated_data/sinus_l0.2_M50_h900_noise0.05_nbsin1.csv --load 1 --save 1 --num_episode 10000 --save_path models_records/A2Csinus2 --load_path models_records/A2Csinus1 --lr 1
python main.py --config config_a2c --df_name generated_data/sinus_l0.2_M50_h900_noise0_nbsin2.csv --load 1 --save 1 --num_episode 10000 --save_path models_records/A2Csinus3 --load_path models_records/A2Csinus2 --lr 0.1
python main.py --config config_a2c --df_name generated_data/sinus_l0.2_M50_h900_noise0.08_nbsin1.csv --load 1 --save 1 --num_episode 10000 --save_path models_records/A2Csinus4 --load_path models_records/A2Csinus3 --lr 0.1

python main.py --config config_a2c --df_name Bitstamp_BTCUSD_2017-2022_minute.csv --load 1 --save 1 --num_episode 50000 --save_path models_records/A2Creal0 --load_path models_records/A2Csinus4 --lr 0.1
python main.py --config config_a2c --df_name generated_data/sinus_l0.2_M50_h900_noise0.05_nbsin1.csv --load 1 --save 1 --num_episode 10000 --save_path models_records/A2Csinus5 --load_path models_records/A2Creal0 --lr 0.3
python main.py --config config_a2c --df_name Bitstamp_BTCUSD_2017-2022_minute.csv --load 1 --save 1 --num_episode 150000 --save_path models_records/A2Creal1 --load_path models_records/A2Csinus5 --lr 0.01