# <em>M<sup>3</sup> QuestionIng</em>: Multi-modal Multi-span Medical Question Answering
*[*Anisha Saha*](https://anisha0325.github.io/), [*Vaibhav Rathore*](https://scholar.google.com/citations?user=VbXpg7MAAAAJ&hl=en), [*Abhisek Tiwari*](https://scholar.google.com/citations?user=aGT5TusAAAAJ&hl=en), [*Akash Ghosh*](https://scholar.google.com/citations?user=NWc6Pw8AAAAJ&hl=en), [*Sai Ruthvik Edara*](https://scholar.google.com/citations?user=prWfqNgAAAAJ&hl=en), [*Sriparna Saha*](https://www.iitp.ac.in/~sriparna/)*  

The repository contains code and dataset for research article titled '<em>M<sup>3</sup> QuestionIng</em>: Multi-modal Multi-span Medical Question Answering' published at ACM Transactions on Computing for Healthcare.

### 📌 Abstract
The growing adoption of AI in healthcare, particularly in preventive care, highlights the critical need for accessibility and precision in Medical Question Answering (MedQA). In recent years, significant efforts have been made to develop multi-span medical question-answering systems, where the answer to a query may span multiple sections or paragraphs of a source document. However, existing systems fall short of aligning with real-world scenarios, where source documents often include both textual and visual content, requiring answers to incorporate images for better comprehension. To address this gap, we propose <em>M<sup>3</sup> QAFrame</em> , a multi-modal, multi-span medical question-answering framework that leverages visual cues to enhance the generation of comprehensive answers drawn from diverse textual and visual spans. The model takes the context, query, and images as input and outputs an answer containing both textual answers and relevant images. The text and image embeddings are processed using a transformer-based architecture to determine the sentence and image relevance. We curate a multi-modal, multi-span medical question-answering dataset, <em>M<sup>3</sup> QuestionIng</em>,  containing queries, medical contexts, associated medical images, and extractive answers. Additionally, each query-answer pair is labeled with user intent and query type to enhance query and context comprehension. Extensive experiments show that our approach consistently outperforms existing methods across various evaluation metrics.


#### : 📄 Full Paper: https://dl.acm.org/doi/abs/10.1145/3820162

###  🗂️ Dataset Access

We provide the dataset for research and academic purposes. To request access to the dataset, please follow the instructions below:

1. **Fill Out the Request Form**: To access the corpus, you need to submit a request through our [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSdb2okCLHNPRYyv6iYrB6smg1yksHVTdl7E6kLh2K4SbnGdUg/viewform?usp=dialog)

2. **Review and Approval**: After submitting the form, your request will be reviewed. If approved, you will receive an email with a link to download the dataset.

3. **Terms of Use**: By requesting access, you agree to:
    - Use the dataset solely for non-commercial, educational, and research purposes.
    - Not use the dataset for any commercial activities.
    - Attribute the creators of this resource in any works (publications, presentations, or other public dissemination) utilizing the dataset.
    - Not disseminate the dataset without prior permission from the appropriate authorities.

🚧 **Code is coming soon!** Stay tuned.  

### Citation:
If you use the dataset or code in your research, please consider citing:
~~~~
@article{sahamulti,
  title={: Multi-modal Multi-span Medical Question Answering},
  author={Saha, Anisha and Rathore, Vaibhav and Tiwari, Abhisek and Ghosh, Akash and Edara, Sai Ruthvik and Saha, Sriparna},
  journal={ACM Transactions on Computing for Healthcare},
  publisher={ACM New York, NY}
}
