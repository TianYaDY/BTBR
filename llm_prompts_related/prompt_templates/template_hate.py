
##########################################################################

PROMPT_TEMPLATE = \
"""
{}

{}

{}

"""



INSTRUCTION_COT = \
f"""

Your task is to help me extract subject-relation-object triples from sentences. Below are the detailed instructions. Thanks for your help!

**Instructions Start

(1) You should carefully read each sentence provided. 
(2) Identify the main subject, the relation (verb or action), and the object in the sentence. The subject is the entity performing the action, the relation is the action itself, and the object is the entity that is receiving the action. 
(3) Your extraction should be clear and explicit, structured as "subject + relation + object". 
(4) Please present each SRO triple in the format: "Subject: XXXX; Relation: XXXX; Object: XXXX". 
(5) After extracting the triple, write it down. It should be in a new line, starting with "####".

Many thanks for your help! I am looking forward to your response!

**Instructions End**

"""



DEMO_TEMPLATE_COT = \
"""

**Example Task No.{} Start**

{}


**Answer**

{}

"""

