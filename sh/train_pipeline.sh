python models/utils/sinus_generator.py --amplitude 50 --period 0.2 --offset 900 --noise 0 --nb_sin 1
python models/utils/sinus_generator.py --amplitude 150 --period 0.4 --offset 1000 --noise 0 --nb_sin 1
python models/utils/sinus_generator.py --amplitude 50 --period 0.2 --offset 900 --noise 0.05 --nb_sin 1
python models/utils/sinus_generator.py --amplitude 50 --period 0.2 --offset 900 --noise 0 --nb_sin 2
python models/utils/sinus_generator.py --amplitude 50 --period 0.2 --offset 900 --noise 0.08 --nb_sin 1

python main.py --df_name generated_data/sinus_l0.2_M50_h900_noise0_nbsin1.csv --load 0 --save 1 --num_episode 100 --save_path sinus0
python main.py --df_name generated_data/sinus_l0.4_M150_h1000_noise0_nbsin1.csv --load 1 --save 1 --num_episode 100 --save_path sinus1 --load_path sinus0
python main.py --df_name generated_data/sinus_l0.2_M50_h900_noise0.05_nbsin1.csv --load 1 --save 1 --num_episode 100 --save_path sinus2 --load_path sinus1
python main.py --df_name generated_data/sinus_l0.2_M50_h900_noise0_nbsin2.csv --load 1 --save 1 --num_episode 100 --save_path sinus3 --load_path sinus2
python main.py --df_name generated_data/sinus_l0.2_M50_h900_noise0.08_nbsin1.csv --load 1 --save 1 --num_episode 100 --save_path sinus4 --load_path sinus3

python main --df_name Bitstamp_BTCUSD_2017-2022_minute.csv --load 1 --save 1 --num_episode 100 --save_path real0 --load_path sinus4

