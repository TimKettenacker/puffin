# puffin

This project is exploring ways to generate ontologies from text (ontology learning). 


Background context of this project:

Creating ontologies from scratch can be hard; doing it by the book, you would usually start off interviewing subject matter experts using competency questions. 
However, this is both time and resource consuming and you would probably end up in a state where you never really know if you have really got a complete grasp of the domain, likely because an expert anticipates you to have some knowledge in this domain or simply implies some facts.
In the industry, you are likely to find some documentation to start with (because some poor intern had to create it...), that can be used as input for NLP tasks.
Ontologies can be used to boost AI applications, because they contain a formal expression of knowledge and can thus steer reasoning, i.e. in a chatbot. This is already applied in some commercially available tools, i.e. IBM Watson.


Explanation of the application:

In this project, I create ontologies for Marvel Movies as a starting point (so far, only "Spider-Man: Far from home" is represented in an ontology).
There are various .py scripts that pass values from one to another, so they would need to stitched together using a bash script or similar.

- "get_plot_outline.py" reads the "plot" part from a given set of wikipedia pages, utilizing beautifulSoup and regex. 
- "extract_pos_construct_triples.py" applies numerous techniques to extract triples from sentences (subject-predicate-object) by combining
  entity extraction from shallow parsing results (sentences are parsed to stanford nlp pipeline, then the part of speect tags are stored in how they related to each other to generate entities and relationships)
  entity extraction from out-of-the-box methods of stanford core (openie which directly extracts s-p-o) 
  this however gives you only the individuals of a to-be ontology; the "data model", the ontology itself, can be aided by 
  extracting involved concepts (i.e. generating "person" out of "Peter Parker") by using stanfords named entity recognition
- "cluster_triples_for_review.py" takes all extracted s-p-o and the concepts and groups similar results so a user can decide in a next step which ones are to keep
- "review_individuals.py" provides a user prompt and presents a group of similar triples; the user can then mark those he wants to keep. Eventually, someone the user has to create the ontology in another tool, using the results as support for mapping both concepts and individuals.
- "semantic_similarity.py" is an encore that mimics chatbot questions and tries to answer them with the ontology loaded into memory (owlready2) and if that fails, returns sentences similar to the input (based on gensim similarity models)


Learnings and takeaways:

A missing link between the sentences could be healed by using co-variances. Many triples start with "He" or "she", this should be replaced by the entity found in the previous sentence(s).
Mind that all of these steps never actually create an ontology themselves, rather, they provide an overview of involved concepts and individuals.
The ontology creation is happening "off screen". Webprotege was used to create an ontology based on the output of "review_individuals.py", also the individuals to the "ontology data model" have to be created in webprotege.
Limit to one domain. Plots usually cover more than one domain. However, AI applications like chatbots really only work proficient in one domain. 
An ontology can steer chatbots, i.e. by entering into a dialogue if a certain node was detected from a NLP component and only some nodes are connected (https://arxiv.org/pdf/1804.04838.pdf).
