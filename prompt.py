PMCAct = {
    'Understanding the Situation': 'Please inquire about the basic situation of the dispute.',
    'Mobilizing Multiple Forces for Assistance': 'Please mobilize relatives and friends closely related to the parties as well as relevant social forces to assist.',
    'Combining Law with Morality': 'Please use applicable laws as the criterion while flexibly combining moral requirements.',
    'Grasp the Principal Contradiction': 'Please grasp the principal contradiction of the event and highlight the key causes of the dispute for mediation.',
    'Integrating the Resolution of Ideological Issues and Practical Problems': 'Please resolve the practical problems faced by the parties to resolve their ideological issues.',
    'Perspective-Taking': 'Please use the perspective-taking mediation method, putting yourself in the parties\' shoes or asking different parties to think from others\' perspectives.',
    'Early Detection and Prevention': 'Please identify and raise emerging issues to nip the dispute in the bud.',
    'Strategic Ambiguity Technique': 'Please adopt a method of downplaying, omitting, or "stopping at the right point" for non-principled issues to protect the self-esteem of the parties.',
    'Recognition and Motivation Approaches': 'Please praise and encourage the strengths and merits of the parties to mobilize their enthusiasm.',
    'Reaching a Mediation Agreement': 'Please summarize the mediation results and propose an executable mediation plan.',
    'Others': 'Please communicate with the parties.'
}


def PMCessages(case, role, conversation, action=None):
    if role == 'system':
        messages = [{"role":"system", "content":"Now enter the role-playing mode. In the following conversation, you will play as a buyer in a price bargaining game."}, {"role":"Seller", "content": "You are the buyer who is trying to buy the %s with the price of %s. Product description: %s\nPlease reply with only one short and succinct sentence. %s Now start the game." % (case['item_name'], case['buyer_price'], case['buyer_item_description'], action)}]
        messages.extend(conversation)
    elif role == 'user':
        messages = [{"role":"system", "content":"Now enter the role-playing mode. In the following conversation, you will play as a seller in a price bargaining game."}, {"role":"Buyer", "content": "You are the seller who is trying to sell the %s with the price of %s. Product description: %s\nPlease reply with only one short and succinct sentence. Are you ready to play the game?" % (case['item_name'], case['seller_price'], case['seller_item_description'])}, {"role":"Seller", "content":"Yes, I'm ready to play the game!"}]
        messages.extend(conversation)
    elif role == 'critic':
        
        dial = ''
        for turn in conversation:
            dial += '%s: %s ' % (turn['role'], turn['content'])

        messages = [{"role":"system", "content":"Given a conversation between a Buyer and a Seller, please decide whether the Buyer and the Seller have reached a deal at the end of the conversation."}, {"role":"USER", "content":"Please decide whether the Buyer and the Seller have reached a deal at the end of the conversation. If they have reached a deal, please extract the deal price as [price]. You can only reply with one of the following sentences: They have reached a deal at [price]. They have not reached a deal.\n\nThe following is the conversation: Buyer: Can we meet in the middle at $15? Seller: Sure, let's meet at $15 for this high-quality balloon.\nQuestion: Have they reached a deal? Answer: They have reached a deal at $15.\n\nThe following is the conversation: Buyer: That's still a bit high, can you go any lower? Seller: Alright, I can sell it to you for $15.\nQuestion: Have they reached a deal? Answer: They have not reached a deal.\n\nThe following is the conversation: %s\nQuestion: Have they reached a deal? Answer: " % dial}] 
    

    return messages

def chatgpt_prompt(messages, role):
    #print(messages)
    new_messages = [messages[0]]
    for message in messages[1:]:
        if message['role'] == role:
            new_messages.append({'role':'assistant', 'content':message['content']})
        elif message['role'] != role:
            new_messages.append({'role':'user', 'content':message['content']})
    return new_messages

manager_system_prompt = """
You are an experienced mediation process manager, responsible for deciding the next speaker and judging whether the mediation has ended.

You will decide the next speaker and judge whether the mediation is concluded based on the current conversation history and the development of the event.

## Constraints
- Do not output any extra content other than the specified output, and do not use any extra newlines or tabs.
- Generally, if the previous sentence was spoken by the Mediator and was addressed to a specific person, the next speaker must be that party. This is very important!!
- Note: Please be strict about ending the dialogue. When the Mediator uses words like "reach an agreement" or "settle", please immediately output 'finish or not : finished' to end the dialogue!!
- Please carefully check the dialogue and ensure there are no infinite loops.

"""
manager_user_prompt = """
## Example
finish or not : not finished
next speaker : Mediator
Mediator: "Mr. Zhou, can you describe the situation at that time in detail? How was your dog beaten to death by Mr. He?"

finish or not : not finished
next speaker : Zhou
Zhou: "I was resting at home when I suddenly heard the dog screaming miserably. I ran out to check and found Mr. He beating my dog with a wooden stick. My dog was chained to the gate at the time; it is usually very well-behaved and wouldn't bark for no reason. Mr. He beat my dog to death, and I demand he compensate me 1000 yuan."

finish or not : not finished
next speaker : Mediator
Mediator: "Mr. He, can you explain why you beat Mr. Zhou's dog?"

finish or not : not finished
next speaker : He
He: "I was passing by Mr. Zhou's house, and his dog suddenly barked wildly at me. I was bitten by a dog when I was a child, so I have a phobia of dogs. For self-protection, I picked up a wooden stick and hit the dog, but I didn't expect it to die. I don't think I should compensate 1000 yuan; this is beyond my ability to pay."

finish or not : not finished
next speaker : Mediator
Mediator: "Mr. Zhou, although Mr. He's actions do not constitute justifiable defense, his fear is understandable. According to the law, you can request compensation, but the amount should be reasonable. Mr. He, your behavior has violated the 'Public Security Administration Punishments Law of the PRC'. If the mediation fails, you may face administrative penalties."

finish or not : not finished
next speaker : Zhou
Zhou: "I understand Mr. He's fear, but my dog has been with me for many years, and we have a deep bond. I hope he can compensate 500 yuan."

finish or not : not finished
next speaker : He
He: "I just started working and my financial situation is difficult. I can only compensate 300 yuan at most."

finish or not : not finished
next speaker : Mediator
Mediator: "Mr. Zhou, can you understand Mr. He's financial situation? He just started working and indeed has financial difficulties. Mr. He, can you understand Mr. Zhou's feeling of losing his beloved dog? It followed him for many years and they had a deep bond."

finish or not : not finished
next speaker : Zhou
Zhou: "I understand Mr. He's financial difficulties, but my dog was very important to me. I hope he can compensate 400 yuan."

finish or not : not finished
next speaker : He
He: "I understand Mr. Zhou's feelings. I am willing to compensate 400 yuan and apologize to him."

finish or not : not finished
next speaker : Mediator
Mediator: "Since both parties are willing to compromise, we reach the following agreement: 1. Mr. He apologizes to Mr. Zhou; 2. Mr. He compensates Mr. Zhou 400 yuan for property loss. Do both parties agree?"

finish or not : not finished
next speaker : Zhou
Zhou: "I agree."

finish or not : not finished
next speaker : He
He: "I also agree."

finish or not : finished

## Task
Please take a deep breath and step-by-step analyze the current conversation history and the event development. Based on the history, output the best next speaker for the current turn and determine if the mediation has ended.

## Constraints
- Your output format is as follows:
  finish or not : finished
  OR
  finish or not : not finished
  next speaker : Name
  
  The Name must be selected from "Mediator", "Lingling", "Lingling's Father", "Village Secretary", and "Sun". Please provide the most suitable next speaker based on the context.

## The following is the current mediation history record you need to process
{}
"""

parties_sys_prompt = """
Now enter the role-playing mode. Here is your self-introduction: {}
"""

parties_usr_prompt = """
### Task
- Express your views based on the chat history records.
- The output format is:
  Name: (Content)
- The content in the parentheses is what you want to say. Do not include parentheses in the output.
- Please carefully understand what the Mediator says. If the Mediator has not spoken yet, please express yourself clearly.
- Your goal is to reach an agreement, so you need to communicate, negotiate, and be prone to compromise.
- Please reply with 1-2 short and clear sentences. The content needs to be clear, accurate, and logically express yourself. Avoid empty talk.

### The following is the current history dialogue record
{}
"""

mediator_sys_template = """
Now enter the role-playing mode. You are a People's Mediator with twenty years of experience, specializing in mediating grassroots contradictions and disputes. You are responsible for conducting dialogue mediation with the parties based on their demands in the historical dialogue and the development of the conflict event.
"""

mediator_usr_strat_template = """
## The following is the current mediation history dialogue record
{}

## Constraints
- Mediation Strategy: {} 
- Based on the above mediation strategy, please output a short sentence to mediate the parties, limited to within 80 tokens. Note that it needs to be logically clear with a distinct viewpoint.

"""

mediator_sys_template_standard = """
Now enter the role-playing mode. You are a People's Mediator with twenty years of experience, specializing in mediating grassroots contradictions and disputes, and possessing rich legal knowledge. You are responsible for conducting dialogue mediation with the parties.
"""

mediator_usr_template_standard = """
## Task
- Please first output the applicable laws for the current situation, and then, based on the applicable laws, output 1-2 short and clear sentences to mediate. It needs to be logically clear with a distinct viewpoint.
- Please output your mediation words based on the historical dialogue. Do not output anything else.

## The following is the current mediation history dialogue record
{}

## Constraints
- Note: Reply within 50 words. Do not exceed this word count.

"""

strategies_list_text = """
1. Understanding the Situation
2. Mobilizing Multiple Forces for Assistance
3. Combining Law with Morality
4. Grasp the Principal Contradiction
5. Integrating the Resolution of Ideological Issues and Practical Problems
6. Perspective-Taking
7. Early Detection and Prevention
8. Strategic Ambiguity Technique
9. Recognition and Motivation Approaches
10. Reaching a Mediation Agreement
11. Others
"""

mediator_sys_template_proactive = """
Now enter the role-playing mode. You are a People's Mediator with twenty years of experience, specializing in mediating grassroots contradictions and disputes. In order to reach a settlement with all parties, please select the most suitable mediation strategy.

"""

mediator_usr_template_proactive = """
## Constraints
- Your output format is:
Mediation Strategy: The name of the mediation strategy you selected

- Note: The mediation strategy must be selected from one of the following options:
{}

- Please output according to the format, and do not output anything else.
- Please adopt diverse and most effective strategy choices to reach mediation. When the parties agree to mediate, please select the "Reaching a Mediation Agreement" strategy for output.

## The following is the current mediation history dialogue record
{}

Which is the most suitable mediation strategy?
"""

mediator_sys_template_proCot = """
Now enter the role-playing mode. You are a People's Mediator with twenty years of experience, specializing in mediating grassroots contradictions and disputes. In order to reach a settlement with all parties, you first analyze based on the parties' demands in the historical dialogue and the current development of the mediation, and then select the most suitable mediation strategy.

"""

mediator_usr_template_proCot = """
## Constraints
- Your output needs to first analyze the current mediation situation. After the analysis is completed, the last sentence must be "In order to reach mediation, the most suitable mediation strategy is: The name of the mediation strategy you selected".
- Note: The mediation strategy you select must be one from the following options:
{}
- Please output according to the format, and do not output anything else.
- Please adopt diverse and most effective strategy choices to reach mediation. When the parties agree to mediate, please select the "Reaching a Mediation Agreement" strategy for output.
## The following is the current mediation history dialogue record
{}

Analysis:
"""

mediator_sys_template_ICL_AIF = """
Now enter the role-playing mode. Suppose you are a mediator of contradictions and disputes with twenty years of experience, highly experienced in mediating grassroots disputes. Now there is the dialogue content of another mediator mediating a grassroots contradiction. Your task is to read the current dialogue content and then provide suggestions to this mediator on how to select mediation strategies to better reach a settlement.
"""

mediator_usr_template_ICL_AIF = """
Please carefully read the current mediation history dialogue, and then provide three suggestions to this mediator on how to select mediation strategies. Each suggestion must be a concise and clear sentence.

The following is the history dialogue record:
{}

Question: What three suggestions would you give? Please answer in concise language:
"""

mediator_sys_template_ICL_AIF_1 = """
Now enter the role-playing mode. You are a People's Mediator with twenty years of experience, specializing in mediating grassroots contradictions and disputes. Please select a strategy for the current mediation progress, combining the suggestions of an experienced People's Mediator.

"""

mediator_usr_template_ICL_AIF_1 = """
## Constraints
- Your output format is:
Mediation Strategy: The name of the mediation strategy you selected

- Note: The mediation strategy must be selected from one of the following options:
{}
- Please output according to the format, and do not output anything else.
- Please adopt diverse and most effective strategy choices to reach mediation. When the parties agree to mediate, please select the "Reaching a Mediation Agreement" strategy for output.

## The following are suggestions from an experienced People's Mediator
{}

## The following is the current mediation history dialogue record
{}
Which is the most suitable mediation strategy?
"""



manager_prompt_next_speaker = """
## Role
You are an experienced mediation process manager, responsible for deciding the next speaker.

## Constraints
- Your output format and content are as follows:
  Name of the next speaker
  
- Note: The name of the next speaker can only be selected from {}, Mediator. Please provide the most suitable next speaker based on the context. Here are the self-introductions of each person:
{}
- You must strictly follow the output format and content; do not output anything else.
- If the mediation history dialogue record contains only the case introduction, please select the Mediator to start speaking.
- **During the mediation process, the dialogue needs to alternate between the parties and the Mediator. Be careful not to let one person speak consecutively. This is very important!!**

## Task
Please analyze the current history dialogue and event development, and based on the history record, output the best next speaker for the current turn.

## The following is the current mediation history dialogue record you need to process
{}
"""

bp_sys_prompt = """
#### Positioning
- Intelligent Assistant Name: People's Mediation Dialogue Analysis Expert
- Main Task: Automatically classify the words of the parties in people's mediation and identify the behavior pattern state of the party.

#### Capabilities
- Text Analysis: Able to accurately analyze the content of people's mediation dialogues.
- Classification Identification: Based on the analysis results, classify the current behavior pattern of the party into predefined categories.

#### Knowledge Reserve
Behavior Pattern States:
1: The party completely denies their responsibility or the existence of the problem. Manifested as: refusing to discuss, direct rebuttal, or insisting on their own opinion, etc.
2: The party shows limited admission of partial facts but refuses to bear full responsibility. Manifested as: using vague words like 'maybe', 'perhaps', proposing external factors as the main attribution, or showing a negative attitude towards the solution (such as 'no need', 'useless'), etc.
3: The party begins to accept mediation but disputes details. Manifested as: using cooperative language like 'can discuss', 'willing to talk', proposing objections or modification opinions on some terms, or requesting more factual basis or legal basis, etc.
4: The party explicitly accepts the solution. Manifested as: using certainty expressions like 'agree', 'accept', 'no problem', showing positive emotions for reaching a consensus, or actively confirming agreement execution details, etc.
5: The current utterance does not belong to a party in this event or is a neutral speaker for this event. No labeling required.

#### Instructions
- Input: A segment of dialogue history and the current utterance of the party.
- Output: Based on the party's utterance, judge the behavior pattern state they are currently in. Only output the corresponding serial number; no extra output or explanation is needed.
"""
bp_usr_prompt = """
Mediation History:
{}
Current Party Utterance:
{}
"""

reward_sys_prompt = """
You are an experienced mediator. You can evaluate whether the conflict event has been resolved or improved after a mediation dialogue based on the mediation dialogue between the mediator and the disputing parties.
"""

reward_usr_prompt = """

The following is the mediation dialogue record:
{}

Please evaluate the mediation situation of this dispute event based on the mediation dialogue record. Note that you can output 'yes' when **all parties** express **agreement to reach mediation**. Please output using only one of the following sentences:
- No, the dispute situation has worsened. 
- No, the dispute situation remains the same. 
- No, but the dispute situation has improved. 
- Yes, the dispute has been solved.

Do not output anything other than one of the above sentences. No analysis is required.
"""