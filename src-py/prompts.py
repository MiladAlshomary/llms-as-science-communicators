answer_focus_question = """
Given a journalistic article and a question, extract all paragraphs from the article that contains answers for the question. Just output the paragraphs and do not add extra explanations to the output.
"""

focus_area_prompt_new = """You task is to  distill 5 different focus area questions from a given press release summary. The goal of these focus areas is to help a journalist direct an interview with the researcher of this paper. Assign each focus question with a given label. 

Labels:
Goal: Research goal;
Fact: A known fact, a statement taken to be true by the author;
Result: The outcome of an experiment;
Hypothesis: A claim proposed by the author;
Method: Experimental method;
Problem: An unresolved or contradictory issue;
Implication: An interpretation of the results,
None: Anything else;

Return the 5 focus area questions, each question in the following format:
[{"Label": "label1", "Focus Question": "focus question"}]"""

gpt_assign_label_new = """You task is to assign every paragraph in the given press release with a corresponding label. The label and definitions are provided below. 

Goal: Research goal;
Fact: A known fact, a statement taken to be true by the author;
Result: The outcome of an experiment;
Hypothesis: A claim proposed by the author;
Method: Experimental method;
Problem: An unresolved or contradictory issue;
Implication: An interpretation of the results,
None: Anything else;

Output the paragraphs as a list of dictionaries where each value in the dictionary is the corresponding label and the paragraph. Like the following format: [{"Label": label, "paragraph":paragraph}]"""


focus_area_prompt = """You task is to  distill 5 different focus areas from a given press release summary. The goal of these focus areas is to help a journalist direct an interview with the researcher of this paper. Assign each focus question with a given label. 

Labels:
Goal: Research goal;
Fact: A known fact, a statement taken to be true by the author;
Result: The outcome of an experiment;
Hypothesis: A claim proposed by the author;
Method: Experimental method;
Problem: An unresolved or contradictory issue;
Implication: An interpretation of the results,
None: Anything else;

Press Release: 
[pr-article]

Return the 5 focus areas in the form of a list: 
[{Label: "label"; "Focus Question": "focus question"}]"""

gpt_assign_label = """You task is to assign every sentence in the given press release with a corresponding label. The label and definitions are provided below. 

Goal: Research goal;
Fact: A known fact, a statement taken to be true by the author;
Result: The outcome of an experiment;
Hypothesis: A claim proposed by the author;
Method: Experimental method;
Problem: An unresolved or contradictory issue;
Implication: An interpretation of the results,
None: Anything else;

Press Release to Tag:
[pr_article]

Output the sentences as a list of dictionaries where each value in the dictionary is the corresponding label and sentence. Like the following format: [{"Label": label, "Sentence":sentence}]
Labeled Sentences:"""


journalist_initial_prompt = """You are a curious journalist who is tasked with the assignment of making a researcher's scientific paper, [sc_title], understandable and engaging to the public. You are given a question to ask the researcher. Phrase the question in a way that is conversational and engaging, while asking the question.

Focus Question: [focus question]
Press Release Article: [pr-article]

Response:"""

journalist_prompt_with_feedback = """You are a curious journalist who is tasked with the assignment of making a researcher's scientific paper understandable and engaging to the public. You are participating in a conversation with a researcher to try and better understand the given focus area, as it relates to the research covered in the paper. You are given a specific question to focus on and the next question that you should ask. Seamlessly integrate this next question into your response. 

Focus Question: [focus question]

Selected PR Article: 
[parsed_pr]

Next Question: [next_question]

Conversation: 
[conversation]"""

researcher_prompt = """You are an enthusiastic researcher who is being interviewed by a science journalist who wants to understand more about your research paper. Your goal is to help the science journalist better understand your work. Be sure to answer the journalist's questions thoroughly and clearly. Explain technical components that an everyday person wouldn't understand. You are given a the title of your paper, the abstract, and the introduction. Use these pieces of information to direct the conversation. Pick up where the conversation left off and write a response to the journalist, be sure to end the conversation in less than 20 turns total. Start wrapping up the discussion around turn 18. To end the conversation, return <STOP>. Only output the researcher's response.

Abstract: 
[abstract]

Introduction:
[introduction]

Number of Turns: [num turns]

Conversation: 
[conversation]"""

llm_journalist_evaluator = """Your job is to evaluate the quality and depth of an conversation between a Journalist and Researcher that is focused on one specific area. The goal of the interview is to help the journalist understand one aspect of the research paper that the Researcher has published. You are also given the gold-standard press release summary that the journalist will need to replicate. You job is to help evaluate how closely the journalist's questions align them with the obtaining all the necessary details to write the press release. 

You will be given the conversation, the focus question, and the press release. You will be asked to evaluate the quality of the Journalist's questions on the given criteria. You should give scores from 0 to 5 for ALIGNMENT, QUALITY, DEPTH, and COMPREHENSION. Half points are allowed. Please refer to the following descriptions:

ALIGNMENT: the Journalist asks thoughtful and probing questions that in line with the given focus area and the gold-standard press release.
QUALITY: Good quality questions are specific, prompt the researcher to provide new information about the material, is related to the overall flow of the conversation, and engages the journalist in a conversational manner.
DEPTH: the Journalist asks questions that expand their own understanding of the subject area. Each question that the researcher asks should further extends the conversation adding new details, new connections, or new perspectives.  
COMPREHENSION: the Journalist demonstrates an understanding of the Researcher's response and asks questions that build off of what the Researcher has said. The Journalist asks clarifying questions when needed. Evaluate the Journalist on their level of understanding of the topic at this current point in the conversation.

Conversation Focus: [focus area]

Gold Standard Press Release: 
[pr-article]

Conversation:
[conversation]

Please present your scores as follows:
QUALITY: [explanation]. GRADE: x/5
DEPTH: [explanation]. GRADE: x/5
COMPREHENSION: [explanation]. GRADE: x/5

NEXT QUESTION: [give 1 specific question that the journalist should ask that is aligned with the press summary and focus area. or if you feel that the focus area has been adequately covered return <STOP>]"""

#########################

baseline_pr_generation = {
    "strategy_name": "baseline_pr_generation",
    "instruction": """
        Please write a press release article to communicate the science presented in the following scientific paper. Your output should be:
        "Press Release Article": "The press release article about the paper"
    """,
    "inputs" : {
        "Scientific paper": "",
    }
}

baseline_pr_generation_cot = {
    "strategy_name": "baseline_pr_generation_cot",
    "instruction": """
        Please write a press release article to communicate the science presented in the following paper.
        Before generating the press release, think step by step about the social impact of the research paper, the innovative aspects of the paper and how it is different from other research on the same topic, and how to communicate the problem, the approach and the results of the paper in a simple and accessible lanuage. Finally output the press release in the following format:
        "Press Release Article": "The press release article about the paper"
    """,
    "inputs" : {
        "Scientific paper": "",
    }
}

baseline_pr_generation_w_conv = {
    "strategy_name": "baseline_pr_generation_w_conv",
    "instruction": """
        Please write a press release article to communicate the science presented in the following paper.
        Before generating the press release, generate a conversation between the paper's author and a journalist where they discuss the social impact of the research paper, the innovative aspects of the paper and how it is different from other research on the same topic. The author explains the paper in a very simple and accessible language to the journalist. Finally output the press release in the following format:
        "Press Release Article": "The press release article about the paper"        
    """,
    "inputs" : {
        "Scientific paper": "",
    }
}

pr_generation_by_conv_summarization = {
    "strategy_name": "pr_generation_by_conv_summarization",
    "instruction": """
        Please write a press release article to communicate the science presented in the following paper. The press-release should summarize the main points in the given conversation. The output should have the following format:
        "Press Release Article": "The press release article about the paper"        
    """,
    "inputs" : {
        "Conversation": "generated-conversation",
    }
}

######################### LLM-based evaluation prompts #############################
factuality_eval_prompt = {
    "strategy_name": "factuality_eval_prompt",
    "instruction": """
        Your job is to evaluate the quality of a conversation between a Journalist and Researcher. The goal of the conversation is to help the journalist understand the Researcher's published paper so they can write a press release on the paper.
        Evaluate the conversation in terms of the provided facts on a scale from 1 to 3:

            Score 1: All information provided in the paper are factually incorrect
            Score 2: Some information provided in the paper are factually incorrect
            Score 3: No information provided in the paper are factually incorrect
            
        First, give reasons for your score and then output the score in the following output format:
            {"reasons": "explain your rating",  "score": "<json integer>"}
        
    """,
    "inputs" : {
        "Conversation": "conversation"
    },
    "scoring_scheme": "3_points"
}

faithfull_eval_prompt = {
    "strategy_name": "faithfull_eval_prompt",
    "instruction": """
        Your job is to evaluate the quality of a conversation between a Journalist and Researcher. The goal of the conversation is to help the journalist understand the Researcher's published paper so they can write a press release on the paper.
        Evaluate the conversation in terms of the accuracy of its information compared to the scientific paper on a scale from 1 to 5:

            Score 1: Everything introduced is inaccurate compared to the scientific paper
            Score 2: The majority of information is inaccurate compared to the scientific paper
            Score 3: Some information is inaccurate compared to the scientific paper
            Score 4: Very little information is inaccurate compared to the scientific paper
            Score 5: All information is accurate compared to the scientific paper
            
        First, give reasons for your score and then output the score in the following output format:
            {"reasons": "explain your rating",  "score": "<json integer>"}
        
    """,
    "inputs" : {
        "Conversation": "conversation",
        "Scientific Paper": "sc-intro",
        "scoring_scheme": "5_points"
    }
}

societal_context_eval_prompt = {
    "strategy_name": "societal_eval_prompt",
    "instruction": """
        Your job is to evaluate the quality of a conversation between a Journalist and a Researcher. The goal of the conversation is to help the journalist understand the Researcher's published paper so they can write a press release on the paper.

        Societal impact refers to the changes that occur in people, communities, and/or environments outside of academia as a result of the research process or findings. These impacts can include conceptual changes, such as new perspectives on socio-ecological challenges, as well as instrumental and capacity-building changes that lead to longer-term social and environmental impacts over time. Stakeholder engagement plays a significant role in driving these societal impacts.

        Evaluate how good the conversation is in placing the paper in its proper societal context on a scale from 1 to 3:

            Score 1: The conversation doesn't mention how the research in the paper impacts society
            Score 2: The conversation discusses the research paper's impact on society in a very general way
            Score 3: The conversation gives a very detailed account of the research paper's impact on society, providing examples and discussing both positive and negative aspects

        When evaluating a conversation under this aspect, the following should be considered:
            - Does the conversation mention how the research in the paper impacts society?
            - Is the discussion of the societal impact brief or extensive?
            - Does the conversation cover the social impact of the paper in a general way, or does it mention things in detail?
            - Does the conversation cover only the positive aspects of the paper, or does it also mention if the research has a negative impact?

            
        First, give reasons for your score and then output the score in the following output format:
            {"reasons": "explain your rating",  "score": "<json integer>"}
        
    """,
    "inputs" : {
        "Conversation": "conversation"
    },
    "scoring_scheme": "3_points"
}

scientific_context_eval_prompt = {
    "strategy_name": "scientific_eval_prompt",
    "instruction": """
        Your job is to evaluate the quality of a conversation between a Journalist and Researcher with the goal of helping the journalist understand the Researcher's published paper so they can write a press release on the paper.

        Scientific context puts the scientific paper in proper context with respect to any other research on the same topic, highlighting the novelty.

        When evaluating a conversation under this aspect, the following should be considered:
            - Does the conversation mention related research on the same topic?
            - Does the conversation mention the related research shortly or in detail?
            - Does the conversation highlight how different or novel this research is in comparison to previous work on the topic?
            - Does the conversation mention how this work helps other scientific research progress on this topic?


        Evaluate how good the conversation is in placing the paper in its proper scientific context on a scale from 1 to 5:
        
            Score 1: The conversation doesn't mention how relevant the paper is to other research on the topic
            Score 2: The conversation mentions how relevant the paper is to other research on the topic in a very general way
            Score 3: The conversation gives a very detailed account of how the paper is grounded in other research on the topic, highlighting the innovation of the paper            
        First, give reasons for your score and then output the score in the following output format:
            {"reasons": "explain your rating",  "score": "<json integer>"}
        
    """,
    "inputs" : {
        "Conversation": "conversation"
    },
    "scoring_scheme": "3_points"
}

clarity_eval_prompt = {
    "strategy_name": "clarity_eval_prompt",
    "instruction": """
        Your job is to evaluate the quality of a conversation between a Journalist and Researcher with the goal of helping the journalist understand the Researcher's published paper so they can write a press release on the paper.

        When evaluating a conversation under this aspect, the following should be considered:

            - Does the conversation contain complex technical concepts that are left unexplained?
            - Does the interviewer ask for explanations to clarify complex aspects of the research?
            - Does the interviewee explain the complexities of their research in an understandable language?
            - Does the interviewee clarify their research via examples, descriptive language, or analogies that help understand their work?
            - Does the interviewee provide background information that makes it easy to understand their work?

        
        Evaluate the clarity and accessibility of the conversation language to the public on a scale from 1 to 5:

            Score 1: The language is very scientific and inaccessible to the public
            Score 2: Many aspects mentioned are very technical and left unexplained
            Score 3: Some aspects mentioned are technical and left unexplained
            Score 4: A few aspects mentioned are technical and left unexplained
            Score 5: The language is understandable by the public, and all technical aspects are clarified

            
        First, give reasons for your score and then output the score in the following output format:
            {"reasons": "explain your rating",  "score": "<json integer>"}
        
    """,
    "inputs" : {
        "Conversation": "conversation",
    },
    "scoring_scheme": "5_points"
}
    

relevancy_eval_prompt = {
    "strategy_name": "relevancy_eval_prompt",
    "instruction": """
        Your job is to evaluate the quality of a conversation between a Journalist and Researcher with the goal of helping the journalist understand the Researcher's published paper so they can write a press release on the paper.
        Evaluate the coverage of the topics and ideas mentioned in the gold standard press release on a scale from 1 to 5:

            Score 1: The gold standard press release is not at all relevant to what is mentioned
            Score 2: Many topics in the gold standard press release are not mentioned
            Score 3: Some topics in the gold standard press release are not mentioned
            Score 4: Few topics in the gold standard press release are not mentioned
            Score 5: All topics in the gold standard press release are mentioned
            
        First, give reasons for your score and then output the score in the following output format:
            {"reasons": "explain your rating",  "score": "<json integer>"}
        
    """,
    "inputs" : {
        "Conversation": "conversation",
        "Gold Standard Press Release": "pr-article"
    },
    "scoring_scheme": "5_points"
}

######################### Press Release Eval Prompts #################################

pr_factuality_eval_prompt = {
    "strategy_name": "factuality_eval_prompt",
    "instruction": """
        Your job is to evaluate the quality of a press release. The goal of the press release is to communicate the science of the paper to the public.
        Evaluate the press release in terms of the provided facts on a scale from 1 to 3:

            Score 1: All information provided in the press release are factually incorrect
            Score 2: Some information provided in the press release are factually incorrect
            Score 3: No information provided in the press release are factually incorrect
            
        First, give reasons for your score and then output the score in the following output format:
            {"reasons": "explain your rating",  "score": "<json integer>"}
        
    """,
    "inputs" : {
        "Press Release": "parsed-pr-article"
    },
    "scoring_scheme": "3_points"
}

pr_faithfull_eval_prompt = {
    "strategy_name": "faithfull_eval_prompt",
    "instruction": """
        Your job is to evaluate the quality of a press release. The goal of the press release is to communicate the science of the paper to the public.
        Evaluate the press release in terms in terms of the accuracy of its information compared to the scientific paper on a scale from 1 to 5:

            Score 1: Everything introduced is inaccurate compared to the scientific paper
            Score 2: The majority of information is inaccurate compared to the scientific paper
            Score 3: Some information is inaccurate compared to the scientific paper
            Score 4: Very little information is inaccurate compared to the scientific paper
            Score 5: All information is accurate compared to the scientific paper
            
        First, give reasons for your score and then output the score in the following output format:
            {"reasons": "explain your rating",  "score": "<json integer>"}
        
    """,
    "inputs" : {
        "Press Release": "parsed-pr-article",
        "Scientific Paper": "sc-intro"
    },
    "scoring_scheme": "5_points"
}

pr_societal_context_eval_prompt = {
    "strategy_name": "societal_eval_prompt",
    "instruction": """
        Your job is to evaluate the quality of a press release. The goal of the press release is to communicate the science of the paper to the public.

        Societal impact refers to the changes that occur in people, communities, and/or environments outside of academia as a result of the research process or findings. These impacts can include conceptual changes, such as new perspectives on socio-ecological challenges, as well as instrumental and capacity-building changes that lead to longer-term social and environmental impacts over time. Stakeholder engagement plays a significant role in driving these societal impacts.

        Evaluate how good the press release is in placing the paper in its proper societal context on a scale from 1 to 3:

            Score 1: The press release doesn't mention how the research in the paper impacts society
            Score 2: The press release discusses the research paper's impact on society in a very general way
            Score 3: The press release gives a very detailed account of the research paper's impact on society, providing examples and discussing both positive and negative aspects

        When evaluating a press release under this aspect, the following should be considered:
            - Does the press release mention how the research in the paper impacts society?
            - Is the discussion of the societal impact brief or extensive?
            - Does the press release cover the social impact of the paper in a general way, or does it mention things in detail?
            - Does the press release cover only the positive aspects of the paper, or does it also mention if the research has a negative impact?

            
        First, give reasons for your score and then output the score in the following output format:
            {"reasons": "explain your rating",  "score": "<json integer>"}
        
    """,
    "inputs" : {
        "Press Release": "parsed-pr-article"
    },
    "scoring_scheme": "3_points"
}

pr_scientific_context_eval_prompt = {
    "strategy_name": "scientific_eval_prompt",
    "instruction": """
        Your job is to evaluate the quality of a press release. The goal of the press release is to communicate the science of the paper to the public.

        Scientific context puts the scientific paper in proper context with respect to any other research on the same topic, highlighting the novelty.

        When evaluating a press release under this aspect, the following should be considered:
            - Does the press release mention related research on the same topic?
            - Does the press release mention the related research shortly or in detail?
            - Does the press release highlight how different or novel this research is in comparison to previous work on the topic?
            - Does the press release mention how this work helps other scientific research progress on this topic?


        Evaluate how good the press release is in placing the paper in its proper scientific context on a scale from 1 to 5:
        
            Score 1: The press release doesn't mention how relevant the paper is to other research on the topic
            Score 2: The press release mentions how relevant the paper is to other research on the topic in a very general way
            Score 3: The press release gives a very detailed account of how the paper is grounded in other research on the topic, highlighting the innovation of the paper            
        First, give reasons for your score and then output the score in the following output format:
            {"reasons": "explain your rating",  "score": "<json integer>"}
        
    """,
    "inputs" : {
        "Press Release": "parsed-pr-article"
    },
    "scoring_scheme": "3_points"
}

pr_clarity_eval_prompt = {
    "strategy_name": "clarity_eval_prompt",
    "instruction": """
        Your job is to evaluate the quality of a press release. The goal of the press release is to communicate the science of the paper to the public.

        When evaluating a press release under this aspect, the following should be considered:

            - Does the press release contain complex technical concepts that are left unexplained?
            - Does the press release provide explanations to clarify complex aspects of the research?
            - Does the press release clarify the research via examples, descriptive language, or analogies that help understand the work?
            - Does the press release provide background information that makes it easy to understand the paper?

        
        Evaluate the clarity and accessibility of the press release language to the public on a scale from 1 to 5:

            Score 1: The language is very scientific and inaccessible to the public
            Score 2: Many aspects mentioned are very technical and left unexplained
            Score 3: Some aspects mentioned are technical and left unexplained
            Score 4: A few aspects mentioned are technical and left unexplained
            Score 5: The language is understandable by the public, and all technical aspects are clarified

            
        First, give reasons for your score and then output the score in the following output format:
            {"reasons": "explain your rating",  "score": "<json integer>"}
        
    """,
    "inputs" : {
        "Press Release": "parsed-pr-article",
    },
    "scoring_scheme": "5_points"
}
    

pr_relevancy_eval_prompt = {
    "strategy_name": "relevancy_eval_prompt",
    "instruction": """
        Your job is to evaluate the quality of a press release. The goal of the press release is to communicate the science of the paper to the public.
        Evaluate the coverage of the topics and ideas mentioned in the gold standard press release on a scale from 1 to 5:

            Score 1: The gold standard press release is not at all relevant to what is mentioned
            Score 2: Many topics in the gold standard press release are not mentioned
            Score 3: Some topics in the gold standard press release are not mentioned
            Score 4: Few topics in the gold standard press release are not mentioned
            Score 5: All topics in the gold standard press release are mentioned
            
        First, give reasons for your score and then output the score in the following output format:
            {"reasons": "explain your rating",  "score": "<json integer>"}
        
    """,
    "inputs" : {
        "Press Release": "parsed-pr-article",
        "Gold Standard Press Release": "pr-article"
    },
    "scoring_scheme": "5_points"
}


######################### New conversation generation prompts #############################

extracting_topic_prompt = {
    "strategy_name": "extract_topics",
    "instruction": """
        Extract a list of all topics that are discussed in the following press release. Each topic should be a phrase of two to three worlds.        
    """,
    "inputs" : {
        "Press Release": "pr-article"
    }
}

# composite_prompt = {
#     "strategy_name": "composite",
#     "instruction": """Please simulate a conversation between a researcher and a journalist regarding the researcher's scientific paper. The goal of the conversation is to gain a deeper understanding of the researcher's scientific paper and communicate its impact to the public through a journalistic report.

#     General Guidelines:
#         1. The conversation should be a maximum of 10 turns
#         2. The Researcher and Journalist both read the scientific paper.
#     Guidelines for Researcher's answers:
#         [researcher-guidelines]
#     Guidelines for the Journalist's questions:
#         [journalist-guidelines]

#     """,
#     "inputs" : {
#         "Scientific paper": "sc-intro",
#     }
# }

composite_prompt = {
    "strategy_name": "composite",
    "instruction": """Please simulate a conversation between a researcher and a journalist regarding the researcher's scientific paper. The goal of the conversation is to gain a deeper understanding of the researcher's scientific paper and communicate its impact to the public through a journalistic report.

    General Guidelines:
        1. The Researcher and Journalist both read the scientific paper.

    Guidelines for Researcher's answers:
        [researcher-guidelines]
    Guidelines for the Journalist's questions:
        [journalist-guidelines]

    """,
    "inputs" : {
        "Scientific paper": "sc-intro",
    }
}

researcher_guidelines = {
    'na': """""", 
    #'young-researcher': """The researcher finds it challenging to simplify their research and convey its impacts to a general audience. Their deep understanding of the subject matter can make it difficult to identify which details are essential for a lay audience and which can be omitted""",
    #'experienced-researcher': """The researcher is excellent at communicating their research in a simple and everyday life language
    #"""
}

journalist_guidelines = {
    'na': "",
    'generic-guidelines': """
    1. The journalist asks questions encouraging the researcher to place their paper in a proper societal and scientific context to the greatest possible degree.
    2. The journalist's questions focus on topics in the paper that are novelty and have unexpected results.
    3. The Journalist's questions follow up on the researcher's answers trying to clarify unexplained technical terms in everyday language.
""",
    'pr-guided': """
    1. The journalist has access to the gold standard journalistic report.
    2. The journalist asks questions to cover topics from the gold standard journalistic report.
    3. The journalist asks follow-up questions to guide the researcher in answering them as they appear in the journalistic report.
    4. The journalist will incrementally deepen the conversation by asking follow-up questions about unexplained technical terms, research methodologies, and the broader impacts of the research.
    """,
    # 'pr-topic-guided':"""
    #     The journalist has to ask questions covering the list of topics provided below
    # """
}