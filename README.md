<div align="center">

# ProMediConv
**Towards Proactive Conversational Mediation Agents for Multi-Party Dispute Resolution**

[Dataset (Coming Soon)] • [Paper (Coming Soon)] • [License](./LICENSE)

</div>

---

### 📢 NEWS
- We have released the codebase and test set of ProMediConv dataset for reference! The full version of the ProMediConv dataset  will be released in this repository after the camera-ready stage.

---

### 📂 Project Structure

```text
ProMediAgent/
├── data/                           # Dataset directory
│   ├── pmc_test.txt                # Dialogue format data (Test Split)
│   └── pmc_test_parties_prompt.txt # Role-play prompts for parties (Test Split)
├── agent.py                        # Policy Planner Model definition
├── env_pmc.py                      # Environment simulation
├── prompt.py                       # Prompt templates
├── sft.py                          # Script for Supervised Fine-Tuning (SFT)
├── run_pmc.py                      # Main entry point for RL training and Evaluation
├── util.py                         # Utility functions
├── LICNECE                         # Licence 
└── README.md                       # Project documentation
```
### Quick Start 🚀

This guide will walk you through the complete workflow of training of ProMediAgent and evaluating the Mediator Agents in ProMediConv.

#### 1. Supervised Fine-Tuning (SFT) for Policy Planner
Initialize the policy planner model with supervised learning on ProMediConv dataset:

```bash
python sft.py --do_train --overwrite_output_dir --per_gpu_train_batch_size=8
```

#### 2. Testing the Post-SFT Policy Planner

```bash
python sft.py --do_eval
```


#### 3. Reinforcement Learning (RL) on Policy Planner
Train the Policy Planner using Reinforcement algorithmn to optimize mediation strategies selection, you can specify the RL checkpoint to load.

```bash
python run_pmc.py --do_train --load_rl_epoch=1
```

#### 4. Testing Mediator Agents in ProMediConv
Benchmarking mediation agents with proactive mediation methods in ProMediConv. Need to specify the proactive methods and backbone model names, here is Qwen-2.5-14B-Instruct / Proactive.

```bash
python run_pmc.py --do_eval --prompt_type="proactive" --system="qwen14b"
```

### DATA INFORMATION 📊

We have 972 cases in total, each of which contains two types of data: `pmc_xxx.txt` and `pmc_xxx_parties_prompt.txt,` where `xxx` are `train`, `valid` or `test` (778:97:97). 

`pmc_xxx.txt`: The dialogue format data.

`pmc_xxx_parties_prompt.txt`: The role-play prompts for all involved dispute parties in all cases.

We trained our Baseline ProMediAgent on `train` dataset. For the extrinsic evalution, we build the evaluation simulation environment based on information in `test` dataset. Now we release the `pmc_test.txt` and  `pmc_test_parties_prompt.txt`for reference under `/Data` .


### DATASET STATISTICS 📈

**BASIC STATISTICS**

| Statistic | Total | Mediator | Party |
|---|----|---|---|
| Average Role Numbers | 4.54 | 1 | 3.54 |
| Number of Cases | 972 | - | - |
| Number of Utterance | 17,277 | 8,774 | 8,503 |
| Average numbers of Utterance per Case | 17.88 | 8.98 | 8.90 |


**DISPUTE CATEGORY (13 Types)**

Dispute Types| Community and Neighborhood | Labor   | Family and Marriage | Estate and Land | Economic Debt | Traffic Accident | Medical and Health   | Public Security and Criminal | Environment Pollution | Campus and Education | Consumer and Tourism | Inheritance | Corporate Management |
|------------|------------------------------|---------|----------------------|-----------------|---------------|------------------|----------------------|-------------------------------|-----------------------|----------------------|-----------------------|-------------|----------------------|
**Proportion** | 19.3%                       | 18.7%   | 18.0%                | 15.9%           | 9.8%          | 7.9%             | 7.3%                 | 5.4%                          | 2.5%                  | 2.2%                 | 3.9%                 | 3.9%        | 1.6%                 |

**MEDIATION STRATEGY SET**
| Strategies                          | Examples                                                                                   | Explanations                                                                 |
|-------------------------------------|-------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| Understanding the Situation          | Hello, Mr. Duan. First, could you please describe what happened at the time in as much detail as possible? | By asking about the basic information of the parties involved, comprehensive support is provided for the mediation process. |
| Mobilizing Multiple Forces for Assistance | I heard that Mr. Sun is a close friend of yours... see if he might be able to offer additional help. | Involving close family, friends, or relevant social groups can help work together to advance the mediation process. |
| Combining Law with Morality         | According to "Regulations on Work-Related Injury Insurance," ..., it should be counted as a work-related injury... | Ensure legality and fairness based on the law, while flexibly applying moral principles to promote problem resolution. |
| Grasp the Principal Contradiction    | The key issue is whether Xinle Company is willing to pay the migrant workers' wages upfront... | Focus on addressing the primary conflict, highlight the key issues in the mediation, and avoid being distracted by secondary problems. |
| Integrating the Resolution of Ideological Issues and Practical Problems | Mr. Li, I understand your concern. The village committee is willing to allocate a piece of land for you to transplant your plants and trees... | Address practical issues to alleviate psychological pressure and create conditions for resolving underlying mental or ideological concerns. |
| Perspective-Taking                  | Xiao Xu, when you were in dire need of help, Xiao Kang was there for you. How about...? | Adopt a perspective of empathy, understand the different needs of the parties involved, and propose acceptable solutions. |
| Early Detection and Prevention      | To prevent potential disputes in the future, we can formalize this agreement in writing, clearly delineating the rights and obligations of both parties. | Identify early signs and potential issues, address them promptly to curb the emergence of disputes, and prevent the escalation of conflicts. |
| Strategic Ambiguity Technique        | Mr. Wu, it's normal for a young couple to have arguments; there's no need to dwell on who's right or wrong... | Minimize or obscure non-principal issues, handling them ambiguously to mediate the conflict while protecting the dignity of the parties involved. |
| Recognition and Motivation Approaches | It's commendable that you recognize this, which shows you're a sensible and reasonable person... | Praise and encourage the parties involved to motivate them, boost their confidence, and promote cooperation in the mediation process. |
| Reaching a Mediation Agreement       | Very well, since both of you are willing to accept mediation, we have reached the following agreement:1..... | Guide the parties to reach an agreement and create an executable written contract that clearly defines their rights and obligations. |
| Others                              | Thank you all for your cooperation. I hope this mediation will help both parties resolve the issue at hand... | Be flexible and adaptable, not limited to using established mediation methods. |




