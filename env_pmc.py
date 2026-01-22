import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch  
import subprocess
import requests
import openai
from openai import OpenAI  
from utils import *
from prompt import *
from prompt import PMCAct
import nltk
import re
import time
import json
import random 

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

class Agent:
    def __init__(self, name, prompt):
        self.name = name
        self.prompt = prompt
    
    def printName(self):
        print("This is {} Agent.\n".format(self.name))
        print("My prompt is", self.prompt[:50])
        
    def speak(self, context):
        while True:
            
            response = get_completion(completion_type="conversation", sys_prompt=parties_sys_prompt.format(self.prompt), usr_prompt=parties_usr_prompt.format(self.name, context))
                        
            if response is not None:
                break
            print("Retrying 'parties speak' generation...")
            # time.sleep(1) 
        # Updated to check for standard English colon
        if ":" in response:
            response = response.split(":", 1)[1]
        return self.name + (": ") + response


class MediatorAgent:
    def __init__(self, args, system_model=None, system_tokenizer=None):
        self.name = "mediator"
        self.args = args
        self.system_model = system_model
        self.system_tokenizer = system_tokenizer

    def printName(self):
        print("I am ", self.name)
        
    def speak(self, context, action):
        current_model_name = self.args.system
        
        if self.args.prompt_type == "ppdpp" or self.args.prompt_type == "ppdpp_nosft":
            print('action = ', action)
            response = get_completion(completion_type="mediator_sft", sys_prompt=mediator_sys_template, usr_prompt=mediator_usr_strat_template.format(context, PMCAct[action]), temperature=0.7, system_model=self.system_model, system_tokenizer=self.system_tokenizer)
            response = response.replace("\n", "")
            return ("Mediator: " + response)

        elif self.args.prompt_type == "standard":
            response = get_completion(
                completion_type="mediator_no_sft", 
                sys_prompt=mediator_sys_template_standard_wis, 
                usr_prompt=mediator_usr_template_standard_wis.format(context), 
                temperature=0.7, 
                system_model=self.system_model, 
                system_tokenizer=self.system_tokenizer,
                model_name=current_model_name  
            )
            
            try:
                if response:
                    response = response.replace("\n", "")
            except AttributeError as e:
                print(f"Error replacing newlines: {e}, response type: {type(response)}")
            if response == "" or response is None:
                response = " "
            return ("Mediator: " + response)

        elif self.args.prompt_type == "proactive":
            response_strat = get_completion(
                completion_type="mediator_no_sft", 
                sys_prompt=mediator_sys_template_proactive, 
                usr_prompt=mediator_usr_template_proactive.format(strategies_list_text, context), 
                temperature=0.7, 
                system_model=self.system_model, 
                system_tokenizer=self.system_tokenizer, 
                max_tokens=200,
                model_name=current_model_name
            )
            
            if response_strat:
                response_cleaned = response_strat.replace(" ", "").replace("\n", "")
                try:
                    # Updated split to English colon
                    strategy_name = response_cleaned.split(":")[1]
                    strat = PMCAct[strategy_name]
                except (KeyError, IndexError):
                    strat = PMCAct["Others"]  
            else:
                strat = PMCAct["Others"]
            
            print("medaition strategy:", strat)

            response_proactive = get_completion(
                completion_type="mediator_no_sft", 
                sys_prompt=mediator_sys_template, 
                usr_prompt=mediator_usr_strat_template.format(context, strat), 
                temperature=0.7, 
                system_model=self.system_model, 
                system_tokenizer=self.system_tokenizer, 
                max_tokens=99999,
                model_name=current_model_name 
            )
            
            response = response_proactive.replace("\n", "") if response_proactive else " "
            return ("Mediator: " + response)

        elif self.args.prompt_type == "proCot":
            response_strat = get_completion(
                completion_type="mediator_no_sft", 
                sys_prompt=mediator_sys_template_proCot, 
                usr_prompt=mediator_usr_template_proCot.format(strategies_list_text, context), 
                temperature=0.7, 
                system_model=self.system_model, 
                system_tokenizer=self.system_tokenizer, 
                max_tokens=200,
                model_name=current_model_name 
            )
            print("Strategy selection result is", response_strat)
            
            if response_strat:
                # More robust extraction of strategy name
                response_cleaned = response_strat.strip()
                # Updated Regex to English expectation
                match = re.search(r"The most suitable mediation strategy is: (.+)", response_cleaned)
                strategy_name = "Others"
                if match:
                    extracted_strategy = match.group(1).strip()
                    if extracted_strategy in PMCAct:
                        strategy_name = extracted_strategy
                    else:
                        for strategy_name_k in PMCAct.keys():
                            if extracted_strategy.startswith(strategy_name_k) or strategy_name_k.startswith(extracted_strategy):
                                strategy_name = strategy_name_k
                                break
                else:
                    print("Strategy selection statement not detected")
            else:
                strategy_name = "Others"

            try:
                strat = PMCAct[strategy_name]
            except KeyError:
                strat = PMCAct["Others"]
            print('strat = ', strat)
            
            response_proactive = get_completion(
                completion_type="mediator_no_sft", 
                sys_prompt=mediator_sys_template, 
                usr_prompt=mediator_usr_strat_template.format(context, strat), 
                temperature=0.7, 
                system_model=self.system_model, 
                system_tokenizer=self.system_tokenizer, 
                max_tokens=99999,
                model_name=current_model_name 
            )
            
            response = response_proactive.replace("\n", "") if response_proactive else " "
            return ("Mediator: " + response)

        elif self.args.prompt_type == "ICL_AIF":
            response = get_completion(
                completion_type="mediator_no_sft", 
                sys_prompt=mediator_sys_template_ICL_AIF, 
                usr_prompt=mediator_usr_template_ICL_AIF.format(context), 
                temperature=0.7, 
                system_model=self.system_model, 
                system_tokenizer=self.system_tokenizer, 
                max_tokens=200,
                model_name=current_model_name
            )
            
            response_strat = get_completion(
                completion_type="mediator_no_sft", 
                sys_prompt=mediator_sys_template_ICL_AIF_1, 
                usr_prompt=mediator_usr_template_ICL_AIF_1.format(strategies_list_text, response, context), 
                temperature=0.7, 
                system_model=self.system_model, 
                system_tokenizer=self.system_tokenizer, 
                max_tokens=200,
                model_name=current_model_name
            )
            
            strategy_name = "Others"
            if response_strat:
                response_cleaned = response_strat.replace(" ", "").replace("\n", "")
                # Updated Regex to English expectation
                match = re.search(r"Mediation Strategy: (.)", response_cleaned)
                if match:
                    first_char = match.group(1)
                    for strategy_name_k in PMCAct.keys():
                        if strategy_name_k.startswith(first_char):
                            strategy_name = strategy_name_k
                            break
                else:
                    print("No matching strategy name found")

            try:
                strat = PMCAct[strategy_name]
            except KeyError:
                strat = PMCAct["Others"]  

            print("medaition strategy:", strat)

            response_proactive = get_completion(
                completion_type="mediator_no_sft", 
                sys_prompt=mediator_sys_template, 
                usr_prompt=mediator_usr_strat_template.format(context, strat), 
                temperature=0.7, 
                system_model=self.system_model, 
                system_tokenizer=self.system_tokenizer, 
                max_tokens=99999999,
                model_name=current_model_name
            )
            
            response = response_proactive.replace("\n", "") if response_proactive else " "
            print('medi_reponse =', response)
            return ("Mediator: " + response)

def get_completion(completion_type, sys_prompt=None, usr_prompt=None, temperature=0.7, frequency_penalty=0, system_model=None, system_tokenizer=None, BP_model=None, BP_tokenizer=None, max_tokens=0, model_name="default"):
        try:
            if completion_type == "mediator_sft":
                print("mediator_sft")
                response = client_mediator.chat.completions.create(
                    model="qwen",
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": usr_prompt},
                    ],
                    temperature=temperature,
                    frequency_penalty=frequency_penalty,
                    stream=False
                )
            
            elif completion_type == "mediator_no_sft":
                print(f"mediator_no_sft (Model: {model_name})")
                
                if model_name == "GLM":
                    messages = [
                        {"role": "system", "content": sys_prompt}, 
                        {"role": "user", "content": usr_prompt},
                    ]
                    
                    input_ids = system_tokenizer.apply_chat_template(
                        messages,  
                        add_generation_prompt=True,
                        tokenize=True
                    )
                    
                    prompt_token_ids = [input_ids]
                    sampling_params = SamplingParams(
                        n=1,
                        max_tokens=max_tokens if max_tokens > 0 else 50,
                        temperature=temperature,
                        seed=42
                    )
                    
                    outputs = system_model.generate(
                        prompt_token_ids=prompt_token_ids,
                        sampling_params=sampling_params
                    )
                    
                    return_response = outputs[0].outputs[0].text
                    return return_response

                else:
                    messages = [
                        {"role": "system", "content": sys_prompt}, 
                        {"role": "user", "content": usr_prompt},
                    ]
                    
                    input_ids = system_tokenizer.apply_chat_template(
                        messages, 
                        add_generation_prompt=True, 
                        tokenize=True
                    )
                    sampling_params = SamplingParams(
                        n=1,              
                        max_tokens=max_tokens if max_tokens > 0 else 50,
                        temperature=temperature,
                        seed=42
                    )
                    prompt_token_ids = [input_ids]
                    outputs = system_model.generate(
                        prompt_token_ids=prompt_token_ids,
                        sampling_params=sampling_params
                    )
                    return_response = outputs[0].outputs[0].text
                    return return_response
            
            elif completion_type == "mediator_no_sft_gpt":
                response = client_gpt3_5.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": usr_prompt},
                    ],
                    temperature=temperature,
                    stream=False
                )
                return_response = response.choices[0].message.content
            elif completion_type == "conversation":
                response = client_deepseek.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": usr_prompt},
                    ],
                    temperature=temperature,
                    frequency_penalty=frequency_penalty,
                    stream=False
                )
                return_response = response.choices[0].message.content
            elif completion_type == "manager":
                response = client_deepseek.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        # Translated role-play instructions
                        {"role": "system", "content": "Now entering role-play mode. You are an experienced mediation process manager responsible for deciding the next speaker."},
                        {"role": "user", "content": usr_prompt},
                    ],
                    temperature=0.0,
                    frequency_penalty=0.1,
                    stream=False
                )
                return_response = response.choices[0].message.content
            
            elif completion_type == "bp_class":
                messages = [
                    {"role":"system", "content": sys_prompt}, 
                    {"role":"user", "content": usr_prompt},
                ]
                
                inputs = BP_tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(BP_model.device)

                outputs = BP_model.generate(
                    **inputs, 
                    max_new_tokens=50,
                    temperature=0.7
                )
                
                return_response = BP_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                return return_response
                
            elif completion_type == "reward":
                response = client_deepseek.chat.completions.create(
                    model="deepseek-chat",
                    messages=[ 
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": usr_prompt},
                    ],
                    temperature=0.0,
                    frequency_penalty=0.0,
                    stream=False
                )
                return_response = response.choices[0].message.content
            else:
                raise ValueError("Invalid completion_type. Must be one of 'mediator_sft', 'mediator_no_sft', 'conversation', 'manager'.")

            return return_response
        except Exception as e:
            print(f"Error during API call: {e}")
            import traceback
            traceback.print_exc()
            return None                 
        
class Env(object):
    def __init__(self, args, dataset, mode, env_system_model=None, env_system_tokenizer=None,env_BP_model=None, env_BP_tokenizer=None):
        
        self.args = args
        self.dataset = dataset[mode]
        self.max_turn = args.max_turn
        self.conversation = []
        #self.bp_matrix = []
        self.cur_conver_step = 0
        self.parties_introduction = ""
        self.mode = mode
        self.agents = []
        
        if mode == "train":
            print("Loading BP model via Transformers (Qwen 14B) on GPU 0...")
            self.BP_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
            self.BP_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-14B-Instruct",
                token="",
                torch_dtype="auto",       
                device_map="auto",        
                max_memory={
                    1: "40GiB",           
                    2: "40GiB"           
                }
            )
            if args.system == "qwen7b":
                self.system_model = LLM(model=args.system_model_path, enforce_eager=True, tensor_parallel_size=2)
                self.system_tokenizer = AutoTokenizer.from_pretrained(args.system_model_path)
            elif args.system == "qwen14b":
                self.system_model = self.BP_model
                self.system_tokenizer = self.BP_tokenizer
            
            elif args.system == "llama3":
                print("Loading Llama 3.1 model via vLLM on GPU 3...")
                model_path = "/model/llama3.1-8B-Instruct"
                
                old_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
                try:
                    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
                    
                    self.system_model = LLM(
                        model=model_path, 
                        enforce_eager=True, 
                        max_model_len=8192, 
                        tensor_parallel_size=1
                    )
                finally:
                    if old_visible_devices:
                        os.environ["CUDA_VISIBLE_DEVICES"] = old_visible_devices
                    else:
                        del os.environ["CUDA_VISIBLE_DEVICES"]
                
                self.system_tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            elif args.system == "GLM":
                print("Loading GLM-4 model via vLLM on GPU 3...")
                model_path = "/model/glm-4-9B-0406"
                
                old_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
                try:
                    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
                    
                    self.system_model = LLM(
                        model=model_path, 
                        enforce_eager=True, 
                        tensor_parallel_size=1, 
                        trust_remote_code=True  
                    )
                finally:
                    if old_visible_devices:
                        os.environ["CUDA_VISIBLE_DEVICES"] = old_visible_devices
                    else:
                        del os.environ["CUDA_VISIBLE_DEVICES"]
                
                self.system_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
        else:
            self.BP_model = env_BP_model
            self.BP_tokenizer = env_BP_tokenizer
            self.system_model = env_system_model
            self.system_tokenizer = env_system_tokenizer
            
        self.reward_dict = {
            'pmc': {
                'worsened': -1.0,
                'same': -0.5,
                'improved': 0.5,
                'solved': 1.0,
            },
        }
        set_random_seed(args.seed)

        
    def reset(self):
        #
        self.cur_conver_step = 0
        if self.mode == 'train':
            self.case = np.random.choice(self.dataset) 
        elif self.mode == 'test':
            self.case = self.dataset[self.test_num]
            
        print('event_type =', self.case['event_type'])
        
        parties_list = []
        self.agents = [] 
        self.parties_introduction = "" 
        
        for parties, prompt in self.case['prompts'].items():
            self.agents.append(Agent(parties, prompt))
            self.parties_introduction = self.parties_introduction + f"{parties}:{prompt}" + '\n'
            parties_list.append(parties)

        self.bp_matrix = {party: [] for party in parties_list}     
        # Translated introduction
        self.parties_introduction = self.parties_introduction + "Mediator: I am the Mediator responsible for this dispute"
        self.agents.append(MediatorAgent(self.args, system_model=self.system_model, system_tokenizer=self.system_tokenizer))
        

        if self.args.data_name == 'pmc':
            # Translated Case Introduction
            self.conversation = "Case Introduction: " + self.case['event_description'] + "\n"
        print(self.conversation)
        return self.conversation

    import subprocess

    def manage(self, context):
        manager_prompt_tmp_next_speaker = manager_oringin_prompt_next_speaker.format(
            self.case['disputing_parties'], self.parties_introduction, context
        )
        while True:
            next_speaker = get_completion(completion_type="manager", usr_prompt=manager_prompt_tmp_next_speaker)
            if next_speaker is not None:
                break
            print("Retrying 'next_speaker' generation...")
            time.sleep(1)  

        return  next_speaker.strip()

    def step(self, policy):
        done = 0
        reward = 0

        print('---------------step:{}-------------'.format(self.cur_conver_step))
        reward_outputs = get_completion(
            completion_type="reward", 
            sys_prompt=reward_sys_prompt, 
            usr_prompt=reward_usr_prompt.format(self.conversation),
            BP_model=self.BP_model, 
            BP_tokenizer=self.BP_tokenizer
        )
        
        reward = self.compute_reward(reward_outputs)# compute reward     

        if self.cur_conver_step == self.max_turn:
            done = -1
            print('--> Maximum number of turns reached !')
            print('turn:', self.cur_conver_step)
            print(self.conversation)
            return self.conversation, reward, done, self.bp_matrix
        
        if self.cur_conver_step != 0:
                

            print('reward output =', reward_outputs)
            print('reward =', reward)

            if reward == 1:
                print('--> Goal completed !')
                print('turn:', self.cur_conver_step)
                print(self.conversation)
                done = 1
                return self.conversation, reward, done, self.bp_matrix
        
        print('--> On-going !')
        
        next_speaker = self.manage(self.conversation)
        print(next_speaker)
        
        for agent in self.agents:
            #print(agent.name)
            if agent.name == next_speaker:
                #print(1)
                if agent.name == "Mediator":
                    if self.args.prompt_type == "ppdpp" or self.args.prompt_type == "ppdpp_nosft":
                        action = policy.select_action(self.conversation)
                        response = agent.speak(self.conversation, action)
                    else:
                        response = agent.speak(self.conversation, "")
                else:
                    response = agent.speak(self.conversation)
                    bp = get_completion(completion_type="bp_class", sys_prompt=bp_sys_prompt, usr_prompt=bp_usr_prompt.format(self.conversation, response), system_model=self.system_model, system_tokenizer=self.system_tokenizer, BP_model=self.BP_model, BP_tokenizer=self.BP_tokenizer)
                    try:
                        self.bp_matrix[agent.name].append(bp)
                    except KeyError as e:
                        # Translated Error Messages
                        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        print(f"KeyError: Party '{agent.name}' is not in bp_matrix keys!")
                        print(f"Current bp_matrix keys: {list(self.bp_matrix.keys())}")
                        print(f"Possible causes: 1. '{agent.name}' was not added to the party list during initialization; 2. Name spelling inconsistency (e.g. extra spaces/typos)")
                        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        raise  # Preserve original error stack
                    print(agent.name)
                    print('bp =', bp)
                print("response: ", response)
                if response:
                    self.conversation  = self.conversation + "\n" + response
                break  

        self.cur_conver_step += 1
        return self.conversation, reward, done, self.bp_matrix

    def compute_reward(self, outputs):
        rewards = []
        try:
            output_list = outputs.split()
        except AttributeError:
            print("Error: 'outputs' is not a string, it's a", type(outputs))
            output_list = []  # Set a default empty list
        #print(outputs)
        for output in output_list:
            #print('reward_output =', output, '\n')
            for key in self.reward_dict[self.args.data_name]:
                if key in output.lower():
                    rewards.append(self.reward_dict[self.args.data_name][key])
                    break
        if len(rewards) == 0:
            reward = 0
        else:
            reward = sum(rewards) / len(rewards)
        print(reward) 
        return reward