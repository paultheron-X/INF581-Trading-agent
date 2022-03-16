#python models/utils/sinus_generator.py --amplitude 50 --period 0.2 --offset 900 --noise 0 --nb_sin 1
#python models/utils/sinus_generator.py --amplitude 150 --period 0.4 --offset 1000 --noise 0 --nb_sin 1
#python models/utils/sinus_generator.py --amplitude 50 --period 0.2 --offset 900 --noise 0.05 --nb_sin 1
#python models/utils/sinus_generator.py --amplitude 50 --period 0.2 --offset 900 --noise 0 --nb_sin 2
#python models/utils/sinus_generator.py --amplitude 50 --period 0.2 --offset 900 --noise 0.08 --nb_sin 1

python main.py --df_name generated_data/sinus_l0.2_M50_h900_noise0_nbsin1.csv --load 0 --save 1 --num_episode 100 --save_path models_records/sinus0 --lr 0.01
python main.py --df_name generated_data/sinus_l0.4_M150_h10_noise0_nbsin1.csv --load 1 --save 1 --num_episode 100 --save_path models_records/sinus1 --load_path models_records/sinus0  --lr 0.01
python main.py --df_name generated_data/sinus_l0.2_M50_h900_noise0.05_nbsin1.csv --load 1 --save 1 --num_episode 100 --save_path models_records/sinus2 --load_path models_records/sinus1 --lr 0.01
python main.py --df_name generated_data/sinus_l0.2_M50_h900_noise0_nbsin2.csv --load 1 --save 1 --num_episode 100 --save_path models_records/sinus3 --load_path models_records/sinus2 --lr 0.01
python main.py --df_name generated_data/sinus_l0.2_M50_h900_noise0.08_nbsin1.csv --load 1 --save 1 --num_episode 100 --save_path models_records/sinus4 --load_path models_records/sinus3 --lr 0.01

python main.py --df_name Bitstamp_BTCUSD_2017-2022_minute.csv --load 1 --save 1 --num_episode 100 --save_path models_records/real0 --load_path models_records/sinus4 --lr 0.001
python main.py --df_name generated_data/sinus_l0.2_M50_h900_noise0.05_nbsin1.csv --load 1 --save 1 --num_episode 100 --save_path models_records/sinus5 --load_path models_records/real0 --lr 0.01
python main.py --df_name Bitstamp_BTCUSD_2017-2022_minute.csv --load 1 --save 1 --num_episode 100 --save_path real1 --load_path models_records/sinus5 --lr 0.001

