python models/utils/sinus_generator.py --amplitude 50 --period 0.2 --offset 900 --noise 0 --nb_sin 1
python models/utils/sinus_generator.py --amplitude 150 --period 0.4 --offset 1000 --noise 0 --nb_sin 1
python models/utils/sinus_generator.py --amplitude 50 --period 0.2 --offset 900 --noise 0.05 --nb_sin 1
python models/utils/sinus_generator.py --amplitude 50 --period 0.2 --offset 900 --noise 0 --nb_sin 2
python models/utils/sinus_generator.py --amplitude 50 --period 0.2 --offset 900 --noise 0.08 --nb_sin 1

python main.py --config dqn_base --df_name generated_data/sinus_l0.2_M50_h900_noise0_nbsin1.csv --load 0 --save 1 --num_episode 10000 --save_path models_records/DQNsinus0 --lr 1
python main.py --config dqn_base --df_name generated_data/sinus_l0.4_M150_h1000_noise0_nbsin1.csv --load 1 --save 1 --num_episode 10000 --save_path models_records/DQNsinus1 --load_path models_records/DQNsinus0  --lr 1
python main.py --config dqn_base --df_name generated_data/sinus_l0.2_M50_h900_noise0.05_nbsin1.csv --load 1 --save 1 --num_episode 10000 --save_path models_records/DQNsinus2 --load_path models_records/DQNsinus1 --lr 1
python main.py --config dqn_base --df_name generated_data/sinus_l0.2_M50_h900_noise0_nbsin2.csv --load 1 --save 1 --num_episode 10000 --save_path models_records/DQNsinus3 --load_path models_records/DQNsinus2 --lr 0.1
python main.py --config dqn_base --df_name generated_data/sinus_l0.2_M50_h900_noise0.08_nbsin1.csv --load 1 --save 1 --num_episode 10000 --save_path models_records/DQNsinus4 --load_path models_records/DQNsinus3 --lr 0.1

python main.py --config dqn_base --df_name Bitstamp_BTCUSD_2017-2022_minute.csv --load 1 --save 1 --num_episode 50000 --save_path models_records/DQNreal0 --load_path models_records/DQNsinus4 --lr 0.1
python main.py --config dqn_base --df_name generated_data/sinus_l0.2_M50_h900_noise0.05_nbsin1.csv --load 1 --save 1 --num_episode 10000 --save_path models_records/DQNsinus5 --load_path models_records/DQNreal0 --lr 0.3
python main.py --config dqn_base --df_name Bitstamp_BTCUSD_2017-2022_minute.csv --load 1 --save 1 --num_episode 150000 --save_path models_records/DQNsreal1 --load_path models_records/DQNsinus5 --lr 0.01

