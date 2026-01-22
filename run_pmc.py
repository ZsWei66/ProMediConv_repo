from env_pmc import Env
from agent import policyPlanner
from utils import *
from itertools import count
from tqdm import tqdm
import argparse
from transformers import BertTokenizer, RobertaTokenizer, BertConfig, RobertaConfig
from fastchat.model import add_model_args
import random
import subprocess
import time
import os
import torch
import numpy as np 

tok = {'bert': BertTokenizer, 'roberta': RobertaTokenizer}
cfg = {'bert': BertConfig, 'roberta': RobertaConfig}


def train(args, config, dataset, filename, tokenizer):
    env = Env(args, dataset, mode='train') # env init
    set_random_seed(args.seed)
    
    policy = None
    if not args.do_eval:
        print("Initializing Policy Planner for Training...")
        policy = policyPlanner(args, config, tokenizer) 
        
        # load policy parameters
        if args.sft_dir is not None and args.prompt_type == 'ppdpp':
            print('Starting loading policy model from {}'.format(args.sft_dir))
            policy.load_model(data_name=args.data_name, filename=args.sft_dir)
            
        if args.load_rl_epoch > 0:
            print('Starting loading RL model in epoch {}'.format(args.load_rl_epoch))
            policy.load_model(data_name=args.data_name, filename=filename, epoch_user=args.load_rl_epoch)
    else:
        print("Eval mode: Skipping Policy Planner initialization (policy set to None).")

    test_performance = []
    if args.do_eval:
        print("Eval mode *************")
        SR16_mean = evaluate_all(args, dataset, policy, filename, 0, env) # eval
        test_performance = [SR16_mean]

    if not args.do_train:
        return
    
    for train_step in range(6, args.max_steps+1):
        SR, AvgT, total_reward = 0., 0., 0.
        loss = torch.tensor(0, dtype=torch.float, device=args.device)

        for i_episode in tqdm(range(args.sample_times), desc='Sampling'):
            print('\n================ New Tuple: {} ===================='.format(i_episode))
            state = env.reset() # init

            epi_reward = 0
            done = 0
            for t in count():   # user dialog
                state, reward, done, bp_matrix = env.step(policy) # history
                epi_reward += reward
                reward = torch.tensor([reward], device=args.device, dtype=torch.float)
                policy.rewards.append(reward)

                if done:
                    if done == 1:
                        SR += 1
                    AvgT += t+1
                    total_reward += epi_reward
                    break
            
            if len(policy.rewards) != 0:
                print('Optimizing model.')
                newloss = policy.optimize_model()
                loss += newloss
            
        enablePrint() 
        print('Loss: {} in epoch_user {}'.format(loss.item()/args.sample_times, args.sample_times))
        print('SR:{}, AvgT:{}, rewards:{} Total epoch_user:{}'.format(SR / args.sample_times,
                    AvgT / args.sample_times, total_reward / args.sample_times, args.sample_times))
        
        if train_step % args.save_num == 0:
            policy.save_model(data_name=args.data_name, filename=filename, epoch_user=train_step)

    print(test_performance)


def evaluate(args, dataset, policy, filename, i_episode, train_env):
    test_env = Env(args, dataset, mode='test') # env init
    set_random_seed(args.seed)
    
    SR, AvgT, total_reward = 0, 0, 0
    SR_turn = [0]* args.max_turn
    
    test_size = len(test_env.dataset)
    selected_count = 10
    selected_indices = random.sample(range(test_size), selected_count)
    
    print(f'Number of evaluation cases: {selected_count} (Total test set size: {test_size})')
    
    test_filename_1 = 'Evaluate-{}-epoch-{}-'.format(args.prompt_type, i_episode)
    record_filename = 'Record-{}-epoch-{}-'.format(args.prompt_type, i_episode) + test_filename_1
    REC_PATH = 'tmp/pmc/eval_result' + record_filename + '.txt'
    
    if not os.path.isdir(TMP_DIR[args.data_name] + '/eval_result/'):
        os.makedirs(TMP_DIR[args.data_name] + '/eval_result/')
    rec_file = open(REC_PATH, 'w')

    for idx in tqdm(selected_indices, desc="Evaluation Progress"):
        print(f'\n================ Test Case {idx} ====================')
        epi_reward = 0
        done = 0
        
        test_env.test_num = idx  
        state = test_env.reset()
        
        for t in count():
            state, reward, done, bp_matrix = test_env.step(policy)
            epi_reward += reward

            if done:
                if done == 1:  
                    SR_turn = [v+1 if i > t else v for i, v in enumerate(SR_turn)]
                    SR += 1
                
                total_reward += epi_reward
                AvgT += t + 1
                rec_file.write('%s\n\n' % str({'dialog': state, 'reward': epi_reward}))
                break
        
    SR_mean = SR / selected_count
    AvgT_mean = AvgT / selected_count
    reward_mean = total_reward / selected_count
    SR_all = [SR_mean, AvgT_mean, reward_mean]
    
    SRturn_all = [v / selected_count for v in SR_turn]
    
    print(f'Success rate per turn: {SRturn_all}')
    print(f'Comprehensive Metrics -- Success Rate: {SR_mean}, Average Turns: {AvgT_mean}, Average Reward: {reward_mean}')
    
    return SR_all

def evaluate_all(args, dataset, policy, filename, i_episode, train_env):
    if 'qwen14b' in [args.system, args.user, args.critic] or \
       'qwen7b' in [args.system, args.user, args.critic] or \
       'llama3' in [args.system, args.user, args.critic] or \
       'GLM' in [args.system, args.user, args.critic]: 
        test_env = Env(args, dataset, mode='test', env_system_model=train_env.system_model, env_system_tokenizer=train_env.system_tokenizer, env_BP_model=train_env.BP_model, env_BP_tokenizer=train_env.BP_tokenizer) 
    else:
        test_env = Env(args, dataset, mode='test') 
    set_random_seed(args.seed)
    
    test_size = len(test_env.dataset)
    print('Test size: ', test_size)

    all_runs_metrics = [] # List of dicts: [{'case_id': 0, 'run_id': 0, 'SR': 1, 'Turns': 5, 'SSR': 3.5, 'BP': 1.2}, ...]

    status_suffix = "_RANDOM" if args.do_random else ""
    status_file = f"eval_status_{args.prompt_type}{status_suffix}_episode_{i_episode}.txt"
    log_file = f"eval_log_{args.prompt_type}{status_suffix}_episode_{i_episode}.txt"

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Evaluation Detail Log ({args.prompt_type}{status_suffix}) - Started at {time.strftime('%X')}\n{'='*50}\n")

    test_filename_1 = 'Evaluate-{}{}-epoch-{}-'.format(args.prompt_type, status_suffix, i_episode)
    record_filename = 'Record-{}{}-epoch-{}-'.format(args.prompt_type, status_suffix, i_episode) + test_filename_1
    REC_PATH = 'tmp/pmc/eval_result' + record_filename + '.txt'
    
    if not os.path.isdir(TMP_DIR[args.data_name] + '/eval_result/'):
        os.makedirs(TMP_DIR[args.data_name] + '/eval_result/')
    rec_file = open(REC_PATH, 'w')
    
    K_RUNS = 1 # use for rollout numbers

    for test_num in tqdm(range(0, test_size), desc="Test Cases"): 
        
        case_metrics = {'SR': [], 'Turns': [], 'SSR': [], 'BP': []}
        
        for k in range(K_RUNS):
            print(f'\n=== Case {test_num+1}/{test_size} | Run {k+1}/{K_RUNS} ===')
            
            test_env.test_num = test_num 
            state = test_env.reset()
            
            epi_reward = 0
            done = 0
            bP = 0
            
            for t in count():
                state, reward, done, bp_matrix = test_env.step(policy)
                epi_reward += reward
                
                if done:
                    # Metric Calculation for this run
                    is_success = 1 if done == 1 else 0
                    turns = t + 1
                    
                    bp_mask = 0
                    bp_val_accum = 0
                    for bp in bp_matrix.values():
                        if len(bp) >= 2 and '5' not in bp:
                            bp_val_accum += int(bp[-1]) - int(bp[0])
                            bp_mask += 1
                    bp_score = bp_val_accum / bp_mask if bp_mask != 0 else 0
                    
                    # Store single run result
                    case_metrics['SR'].append(is_success)
                    case_metrics['Turns'].append(turns)
                    case_metrics['SSR'].append(epi_reward)
                    case_metrics['BP'].append(bp_score)
                    
                    rec_file.write(f"Case {test_num} Run {k}: " + '%s\n\n' % str({'dialog':state, 'reward':epi_reward}))
                    
                    # Log single run details
                    with open(log_file, 'a', encoding='utf-8') as f_log:
                        f_log.write(f"Case: {test_num+1} | Run: {k+1} | SR: {is_success} | Reward: {epi_reward:.2f} | BP: {bp_score:.2f} | Turns: {turns}\n")
                    break
        
        for k in range(K_RUNS):
             all_runs_metrics.append({
                 'SR': case_metrics['SR'][k],
                 'Turns': case_metrics['Turns'][k],
                 'SSR': case_metrics['SSR'][k],
                 'BP': case_metrics['BP'][k]
             })
        
        total_runs_so_far = len(all_runs_metrics)
        if total_runs_so_far > 0:
            cur_sr = np.mean([m['SR'] for m in all_runs_metrics])
            cur_avgt = np.mean([m['Turns'] for m in all_runs_metrics])
            cur_ssr = np.mean([m['SSR'] for m in all_runs_metrics])
            cur_bp = np.mean([m['BP'] for m in all_runs_metrics])
            
            # Calculate Standard Deviations (Global)
            cur_sr_std = np.std([m['SR'] for m in all_runs_metrics])
            cur_avgt_std = np.std([m['Turns'] for m in all_runs_metrics])
            cur_ssr_std = np.std([m['SSR'] for m in all_runs_metrics])
            cur_bp_std = np.std([m['BP'] for m in all_runs_metrics])

            print(f"Global Stats ({total_runs_so_far} runs): SR={cur_sr:.4f}, AvgT={cur_avgt:.4f}, SSR={cur_ssr:.4f}, BP={cur_bp:.4f}")
            
            try:
                with open(status_file, 'w', encoding='utf-8') as f_stat:
                    f_stat.write(f"======== Random Strategy Analysis Monitor ({args.prompt_type}) ========\n")
                    f_stat.write(f"Time:         {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f_stat.write(f"Mode:         {'Random (k=5)' if args.do_random else 'Standard'}\n")
                    f_stat.write(f"Case Prog:    {test_num + 1} / {test_size}\n")
                    f_stat.write(f"Total Runs:   {total_runs_so_far}\n")
                    f_stat.write("-" * 46 + "\n")
                    f_stat.write(f"{'Metric':<10} | {'Mean':<10} | {'Std Dev':<10}\n")
                    f_stat.write("-" * 46 + "\n")
                    f_stat.write(f"{'SR':<10} | {cur_sr:<10.4f} | {cur_sr_std:<10.4f}\n")
                    f_stat.write(f"{'AvgT':<10} | {cur_avgt:<10.4f} | {cur_avgt_std:<10.4f}\n")
                    f_stat.write(f"{'SSR':<10} | {cur_ssr:<10.4f} | {cur_ssr_std:<10.4f}\n")
                    f_stat.write(f"{'BP':<10} | {cur_bp:<10.4f} | {cur_bp_std:<10.4f}\n")
                    f_stat.write("-" * 46 + "\n")
            except Exception as e:
                print(f"Write status file error: {e}")

    final_sr = np.mean([m['SR'] for m in all_runs_metrics])
    final_avgt = np.mean([m['Turns'] for m in all_runs_metrics])
    final_ssr = np.mean([m['SSR'] for m in all_runs_metrics])
    final_bp = np.mean([m['BP'] for m in all_runs_metrics])
    
    final_sr_std = np.std([m['SR'] for m in all_runs_metrics])
    final_avgt_std = np.std([m['Turns'] for m in all_runs_metrics])
    final_ssr_std = np.std([m['SSR'] for m in all_runs_metrics])
    final_bp_std = np.std([m['BP'] for m in all_runs_metrics])

    print('Test evaluation saved successfully!')
    SR_all = [final_sr, final_avgt, final_ssr, final_ssr, final_bp] # Return formatting
    return SR_all

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-seed', type=int, default=25, help='Random seed.')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs.')
    parser.add_argument('--epochs', '-me', type=int, default=50000, help='The number of RL training epochs')
    parser.add_argument('--gamma', type=float, default=0.999, help='Reward discount factor.')
    parser.add_argument('--learning_rate', type=float, default=1e-6, help='Learning rate.')

    parser.add_argument('--data_name', type=str, default='pmc')
    
    parser.add_argument('--system', type=str, default='GLM', choices=['qwen7b','qwen14b','GLM', 'ChatGpt', 'llama3'],
                        help='One of {qwen7b,qwen14b,GLM, ChatGpt, llama3}.')
    
    parser.add_argument('--sft_dir', default='/sft/pmc/bert/best_checkpoint', 
                        type=str, help="Pretrain model path.")
    parser.add_argument('--max_turn', type=int, default=20, help='Max conversation turn')
    parser.add_argument('--mode', type=str, default='train', help='The mode in [train, test]')
    parser.add_argument('--load_rl_epoch', type=int, default=0, help='Load agent from specific epoch')


    parser.add_argument("--cache_dir", default='/storage_fast', type=str, help="The cache directory.")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--BP_model_path", type=str, default="/model/qwen2_5/Qwen2.5-14B-Instruct")
    parser.add_argument("--system_model_path", type=str, default="/model/qwen2_5/Qwen2.5-14B-Instruct")
    parser.add_argument("--model_name", type=str, default="roberta")
    parser.add_argument("--model_name_or_path", default='/model/Roberta', type=str, help="Model name or path")

    parser.add_argument("--do_lower_case", action='store_false', help="Set this flag if you are using an uncased model.")

    parser.add_argument('--max_steps', type=int, default=10, help='Max training steps')
    parser.add_argument('--sample_times', type=int, default=100, help='The epoch of sampling')
    parser.add_argument('--eval_num', type=int, default=1, help='The number of steps to evaluate RL model and metric')
    parser.add_argument('--save_num', type=int, default=1, help='The number of steps to save RL model and metric')

    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval.")

    parser.add_argument('--prompt_type', type=str, default='proactive', choices=['ppdpp','ppdpp_nosft','standard','proactive','proCot',"ICL_AIF"],
                        help='One of {ppdpp,standard,proactive,proCot,ICL_AIF}.')
    add_model_args(parser)
    args = parser.parse_args()
    
    print(args.device)
    print('data_set:{}'.format(args.data_name))

    dataset = load_pmc_env_dataset(args.data_name)
    filename = '{}-{}'.format(args.data_name,args.sft_dir)

    config = None
    tokenizer = None

    if not args.do_eval:
        print(f"Loading {args.model_name} from {args.model_name_or_path} ...")
        config = cfg[args.model_name].from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
        tokenizer = tok[args.model_name].from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case, cache_dir=args.cache_dir)
    else:
        print(f"Skipping loading {args.model_name} because do_eval is True.")

    if args.sft_dir and not os.path.exists(args.sft_dir):
        print("No SFT model found, randomly initializing policy model")
        args.sft_dir = None

    train(args, config, dataset, filename, tokenizer)

if __name__ == '__main__':
    main()