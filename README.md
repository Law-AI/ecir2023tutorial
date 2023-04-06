# Legal IR and NLP: the History, Challenges, and State-of-the-Art
This repository contains all detailed information and resources for our tutorial at [ECIR 2023](https://ecir2023.org/), held at Dublin, Ireland (April 2023).

# Abstract
Artificial Intelligence (AI), Machine Learning (ML), Information Retrieval (IR) and Natural Language Processing (NLP) are transforming the way legal professionals and law firms approach their work. The significant potential for the application of AI to Law, for instance, by creating computational solutions for legal tasks, has intrigued researchers for decades. This appeal has only been amplified with the advent of Deep Learning (DL). It is worth noting that working with legal text is far more challenging than in many other subdomains of IR/NLP, mainly due to factors like lengthy documents, complex language and lack of large-scale datasets. In this tutorial, we shall introduce the audience to the nature of legal systems and texts, and the challenges associated with processing legal documents. We shall then touch upon the history of AI and Law research, and how it has evolved over the years from rudimentary approaches to DL techniques. There will also be a brief introduction into the recent, state-of-the-art research in general domain IR and NLP. We shall then discuss in more detail about specific IR/NLP tasks in the legal domain and their solutions, available tools and datasets, as well as the industry perspective. This will be followed by a hands-on coding/demo session, which is likely to be of great practical benefit to the attendees.

# Tutorial Outline
**Part** | **Topic** | **Presenter** | **Link to Slides**
--- | --- | --- | ---
1 | Background on legal text | Saptarshi Ghosh | -
2 | Brief history of AI-Law and important milestones | Jack G. Conrad | -
3 | Background on NLP and IR | Pawan Goyal | -
4 | State-of-the-art survey | Debasis Ganguly, Paheli Bhattacharya and Kripabandhu Ghosh | -
5 | Industry perspective | Jack G. Conrad | -
6 | Future directions, advent of LLMs and explainability | Jack G. Conrad, Kripabandhu Ghosh and Saptarshi Ghosh | -
7 | Hands-on coding | Debasis Ganguly, Paheli Bhattacharya, Shounak Paul and Shubham Kumar Nigam | -

# Task-specific Resources
This section contains resources for different automation tasks in the legal domain

## Legal Named Entity Recognition

This task aims to identify different entities in legal documents. Entities may be classified into different groups that have different legal meanings, such as the parties (appellants, respondents), lawyers, judges and so on.

+ <a href="https://github.com/elenanereiss/Legal-Entity-Recognition"> A Dataset of German Legal Documents for Named Entity Recognition </a> (<a href="https://aclanthology.org/2020.lrec-1.551.pdf">Lietner et al., 2020</a>)
+ <a href="https://github.com/Legal-NLP-EkStep/legal_NER"> Named Entity Recognition in Indian court judgments </a> (<a href="https://arxiv.org/pdf/2211.03442.pdf">Kalamkar et al., 2022</a>)

## Legal Summarization

The task of summarization in the legal domain aims to generate a gist of the entire case document, either in extractive fashion (selecting the most important sentences) or abstractive fashion (similar to summaries written by humans).

+ <a href="https://github.com/Law-AI/summarization"> Legal Case Document Summarization: Extractive and Abstractive Methods and their Evaluation </a> (<a href="https://arxiv.org/pdf/2210.07544.pdf">Bhattacharya et al., 2022</a>)
+ <a href="https://evasharma.github.io/bigpatent/"> BIGPATENT: A Large-Scale Dataset for Abstractive and Coherent Summarization </a> (<a href="https://aclanthology.org/P19-1212.pdf">Sharma et al., 2019</a>)


## Legal Judgment Prediction

Broadly speaking, this task aims to determine the outcomes of court cases. In many settings, this may be composed of several sub-tasks, which are addressed in the forthcoming sections.

+ <a href="https://github.com/koc-lab/law-turk"> Natural language processing in law: Prediction of outcomes in the higher courts of Turkey </a> (<a href="https://www.sciencedirect.com/science/article/pii/S0306457321001692?via%3Dihub">Mumcuoglu et al., 2021</a>)
+ <a href="https://pub.cl.uzh.ch/wiki/public/pacoco/swiss_legislation_corpus"> Building corpora for the philological study of Swiss legal texts </a> (<a href="https://www.zora.uzh.ch/id/eprint/54761/">Hofler et al., 2011</a>)
+ <a href="https://github.com/Exploration-Lab/CJPE"> ILDC for CJPE: Indian Legal Documents Corpus for Court Judgment Prediction and Explanation </a> (<a href="https://aclanthology.org/2021.acl-long.313.pdf">Malik et al., 2021</a>)
+ <a href="https://github.com/masha-medvedeva/ECtHR_crystal_ball"> Judicial Decisions of the European Court of Human Rights: Looking into the Crystal Ball </a> (<a href="http://martijnwieling.nl/files/Medvedeva-submitted.pdf">Medvedeva et al., 2018</a>)
+ <a href="https://github.com/thunlp/CAIL"> CAIL2018: A Large-Scale Legal Dataset for Judgment Prediction </a> (<a href="https://arxiv.org/pdf/1807.02478.pdf">Xiao et al., 2018</a>)

## Identifying Legal Articles and Charges

Often considered a sub-task of Legal Judgment Prediction, this task aims to identify the relevant legal articles and charges given the facts of a case.

+ <a href="https://github.com/Law-AI/LeSICiN"> LeSICiN: A Heterogeneous Graph-based Approach for Automatic Legal Statute Identification from Indian Legal Documents </a> (<a href="https://ojs.aaai.org/index.php/AAAI/article/view/21363">Paul et al., 2022</a>)
+ <a href="https://github.com/IntelligentLaw/HMN"> Hierarchical Matching Network for Crime Classification </a> (<a href="https://dl.acm.org/doi/pdf/10.1145/3331184.3331223">Wang et al., 2019</a>)
+ <a href="https://github.com/Law-AI/automatic-charge-identification"> Automatic Charge Identification from Facts: A Few Sentence-Level Charge Annotations is All You Need </a> (<a href="https://aclanthology.org/2020.coling-main.88.pdf">Paul et al., 2020</a>)
+ <a href="https://github.com/nlp208/legal_attention"> Charge Prediction with Legal Attention </a> (<a href="https://link.springer.com/chapter/10.1007/978-3-030-32233-5_35">Bao et al., 2019</a>)

## Semantic Segmentation / Rhetorical Role Labeling

Court case documents are composed of several functional parts such as Facts, Arguments, Ruling, etc. which may not be clearly demarcated. This task aims to automate the process of segmenting a court case document into these parts.

+ <a href="https://github.com/Law-AI/semantic-segmentation"> Identification of Rhetorical Roles of Sentences in Indian Legal Judgments </a> (<a href="https://arxiv.org/pdf/1911.05405.pdf">Bhattacharya et al., 2019</a>)
+ <a href="https://datasets.doctrine.fr/"> The French Court Decision Structure dataset — FCD12K </a>

## Pre-trained Language Models for the Legal Domain

Recently there have been many efforts to pre-train large, transformer-based language models for the legal domain, which have been adapted to many down-stream end tasks with spectacular efficiency.

+ <a href="https://huggingface.co/nlpaueb/legal-bert-base-uncased"> LEGAL-BERT: The Muppets straight out of Law School </a> (<a href="https://aclanthology.org/2020.findings-emnlp.261.pdf">Chalkidis et al., 2020</a>)
+ <a href="https://huggingface.co/zlucia/legalbert"> When Does Pretraining Help? Assessing Self-Supervised Learning for Law and the CaseHOLD Dataset of 53,000+ Legal Holdings </a> (<a href="https://arxiv.org/pdf/2104.08671v3.pdf">Zheng et al., 2021</a>)
+ <a href="https://huggingface.co/pile-of-law/legalbert-large-1.7M-1"> Pile of Law: Learning Responsible Data Filtering from the Law and a 256GB Open-Source Legal Dataset </a> (<a href="https://arxiv.org/pdf/2207.00220.pdf">Henderson et al., 2022</a>)
+ <a href="https://huggingface.co/law-ai/InLegalBERT"> Pre-training Transformers on Indian Legal Text </a> (<a href="https://arxiv.org/pdf/2209.06049.pdf">Paul et al., 2022</a>)

## General Resources / Benchmarks

This is a miscellaneous list of other resources.

+ <a href="https://github.com/coastalcph/lex-glue"> LexGLUE: A Benchmark Dataset for Legal Language Understanding in English </a> (<a href="https://aclanthology.org/2022.acl-long.297.pdf">Chalkidis et al., 2022</a>)
+ <a href="https://github.com/Liquid-Legal-Institute/Legal-Text-Analytics"> Liquid Legal Institute Repository on Legal Text Analytics </a>

# Presenters

+ [Debasis Ganguly](https://gdebasis.github.io/), Lecturer (Assistant Professor), School of Computing Science, University of Glasgow, Glasgow, Scotland
+ [Jack G. Conrad](http://www.conradweb.org/~jackg/), Director of Applied Research, Thomson Reuters Labs, Minneapolis, MN  USA
+ [Kripabandhu Ghosh](https://www.iiserkol.ac.in/web/en/people/faculty/cds/kripaghosh), Assistant Professor, Department of Computational & Data Sciences, IISER Kolkata, West Bengal, India
+ [Saptarshi Ghosh](http://cse.iitkgp.ac.in/~saptarshi), Assistant Professor, Department of Computer Science & Engineering, IIT Kharagpur, West Bengal, India
+ [Pawan Goyal](http://cse.iitkgp.ac.in/~pawang), Associate Professor, Deptt. of Computer Science & Engineering, IIT Kharagpur, West Bengal, India
+ [Paheli Bhattacharya](https://sites.google.com/site/pahelibh/)
+ [Shubham Kumar Nigam](https://sites.google.com/view/shubhamkumarnigam), Senior Research Fellow, Department of Computer Science & Engineering, IIT Kanpur, Uttar Pradesh, India
+ [Shounak Paul](https://sites.google.com/view/shounakpaul95), Senior Research Fellow, Department of Computer Science & Engineering, IIT Kharagpur, West Bengal, India
