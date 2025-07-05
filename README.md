# End-to-end Text Summarization using HuggingFace- Development to Deployment

## Description of the Project:
The Text-Summarizer-Project 🧠 is an advanced NLP system designed to perform automatic text summarization using state-of-the-art deep learning models. With the exponential growth of unstructured textual data, extracting salient information efficiently has become a core challenge in natural language processing.

This project leverages transformer-based architectures to build a general-purpose summarization pipeline capable of handling diverse text sources such as research papers, articles, and news content 📄. It utilizes tokenization, attention mechanisms, and sequence-to-sequence learning to identify and preserve key semantic components while generating concise summaries.

The system is built with scalability and modularity in mind, making it suitable for both real-time and batch summarization tasks. Its architecture supports fine-tuning on custom datasets, enabling domain-specific summarization use cases with high accuracy and relevance.


⚙️ Key Features and Functionality:

- Text Preprocessing 🧹: Cleans input by removing noise, punctuation, and stopwords to improve summary quality.

- Sentence Extraction 🧠: Applies NLP techniques to identify sentences that best represent the core ideas.

- Semantic Understanding 🔍: Uses semantic analysis to grasp context and extract the most relevant content.

- Summarization Techniques 📝: Supports both extractive (selecting key sentences) and abstractive (paraphrasing and rewriting) methods.

- Length Control 📏: Allows users to customize the length of the summary — short or detailed, based on need.

- User Interface 💻: Intuitive and user-friendly UI for inputting text and viewing results, with features like text highlighting and optional source citation.

## 🌟  **Benefits and Impact**
The Text-Summarizer-Project brings a range of practical benefits and real-world applications:

⏱️ Time-Saving: Quickly delivers concise summaries of lengthy content, helping users digest information more efficiently.

📚 Research & Knowledge Management: Ideal for researchers, students, and professionals to extract key points from academic papers, reports, and large documents.

📰 Content Curation: Supports journalists, content creators, and publishers in generating short summaries of articles, blogs, and online content—boosting engagement and simplifying content discovery.

🗣️ Language Learning: Assists language learners in practicing comprehension by summarizing complex or foreign language texts.

🔎 Information Retrieval: Can be integrated into search engines or IR systems to present quick, relevant summaries alongside results, improving the overall user experience.

In essence, the project is designed to transform how we consume and process information, enabling fast, accurate, and accessible text summarization for a wide range of users.

## 📚 Dataset and Model Description
For fine-tuning, we utilize pegasus-cnn_dailymail by Google, a powerful transformer-based model pre-trained on 200k+ news articles for abstractive summarization 🧠. The dataset consists of article-summary pairs from CNN and DailyMail, and a typical sample looks like this:

![image](https://github.com/user-attachments/assets/21380172-436a-4317-b143-069fdbab77fc)


We fine-tune the final layer of our model using the Samsum dataset 🗣️, which contains over 16,000 dialogue-summary pairs focused on casual, real-life conversations. This helps the model better understand and summarize human dialogue.
A sample from the dataset is shown below:

![image](https://github.com/user-attachments/assets/42bb9271-12e4-4804-8eef-61454e23fb6b)

### Data Fields
- dialogue: text of dialogue.
- summary: human written summary of the dialogue.
- id: unique id of an example.

### Data Splits
- train: 14732
- val: 818
- test: 819

## 🧠 Model Information: PEGASUS
PEGASUS is a cutting-edge abstractive text summarization model developed by Google AI. Built on the transformer architecture, PEGASUS has been pre-trained on a massive dataset of text (and optionally, code), enabling it to generate summaries that are both fluent and highly informative.

🔍 How It Works
PEGASUS is trained using a variant of masked language modeling, where key sentences are masked and the model learns to reconstruct them. This unique pretraining objective closely mimics the summarization task, helping PEGASUS understand context, syntax, and semantics more effectively than traditional models.

💡 Key Features:
Transformer-based architecture proven effective for NLP tasks

Pretrained on large-scale corpora for broad language understanding

Generates fluent, coherent, and content-rich summaries

Outperforms many existing models on summarization benchmarks

Applicable to a wide range of domains:

📰 News article summarization

📄 Research paper summarization

💻 Code/documentation summarization

PEGASUS stands as a powerful and versatile tool for modern summarization tasks. Its performance and flexibility make it a strong choice for applications in academia, journalism, software development, and more.

## Project Flow :

![image](https://github.com/user-attachments/assets/733acb32-d053-40d8-9735-c766ebe08b69)

## Project Setup:
### Git Repo
-Created a GitHub repo to ensure version control of our code. 

🛠️ Code Setup
To streamline project initialization, I started by creating a template.py file that automatically generates the required folder structure when executed. I also added custom logging to track execution flow and help visualize exceptions 🔍.

I then created and activated a virtual environment named textSum using Anaconda and listed all dependencies in requirements.txt 📦.

If I decide to publish the model as an installable package (e.g., on PyPI), I can create a setup.py file to manage metadata and install local modules by referencing file constructors.

📂 Utilities Module
In the utils folder, I wrote a common.py file to store reusable utility functions, such as:

create_directories()

read_yaml()

I also handled centralized exception management within this file.

Key imports include:

ConfigBox 🧩: Simplifies access to nested dictionary values.

ensure_annotations ✅: A Python decorator that enforces type-checking for function parameters.

🧾 Custom Logging
I implemented a logging system in the logging folder for consistent and easy-to-read logs across the application.

🔬 Experiment
Before implementing modular code, I performed initial experiments to visualize the data, tune hyperparameters, and record evaluation metrics such as ROUGE scores 📊. This helped in understanding the dataset better and establishing performance baselines for future improvements.

⚙️ Development Part 1: ML Model Pipeline
The development of the model was carried out in five key stages:

📥 Data Ingestion
I imported the Samsum training dataset hosted on GitHub.

✅ Data Validation
I ensured that the data was correctly structured into train, validation, and test directories.

🔄 Data Transformation
I used the PEGASUS tokenizer to convert raw text into model-ready input features — adding attention masks and labels for training.

🎯 Model Training
I trained the model by first declaring all necessary hyperparameters and configurations in a params.yaml file.

📊 Model Evaluation
Finally, I evaluated the model’s performance using ROUGE scores, a standard metric for summarization quality.


## Workflows 

1. config.yaml 
2. params.yaml 
3. config entity 
4. configuration manager
5. Update the components - Data Ingestion, Data Transformation, Model Trainer, Model Evaluation 
6. Pipeline (Training and Prediction)
7. Frontend - API's (Training and Batch Prediction)

![image](https://github.com/user-attachments/assets/5ceaad43-059c-4b36-876e-098b30dbeb99)

🖥️ Development Part 2: Prediction Pipeline & User Interface
In the second and final step of development, I built a prediction pipeline and integrated it with a user interface using FastAPI.

🚀 FastAPI Integration
I used FastAPI to spin up a lightweight, local web app that enables interaction with the model via HTTP requests.

📦 Prediction Pipeline
Using the pipeline utility from the Transformers library, I loaded the trained PEGASUS model and used it to perform text summarization on new input data.

🛠️ API Routes in app.py
I defined two main endpoints to handle client-server communication:

/train 🔧

Triggers an HTTP request to execute main.py

Runs the 5-stage model development process to train and save the summarization model on the server

/predict ✨

Accepts an HTTP POST request with dialogue text as input

Uses the trained model to generate a summary

Returns the summarized output as a response to the client

![website](https://github.com/user-attachments/assets/1c7eace9-2cbd-487c-bb90-12691d0b49c3)

☁️ Deployment on AWS using Docker, ECR, EC2 & GitHub Actions
In the final stage of the project, I deployed the summarization model's UI to the internet using AWS services and GitHub Actions for automation.

🐳 Docker & EC2
I used Docker, installed on an AWS EC2 Ubuntu instance, to build a container image from the project using a predefined Dockerfile.

📦 AWS ECR (Elastic Container Registry)
The Docker image is pushed to AWS ECR, a secure container image registry managed by AWS. IAM roles and policies are configured to allow EC2 instances and GitHub Actions to authenticate and interact with ECR.

⚙️ GitHub Actions CI/CD Workflow
A custom GitHub Actions workflow automates the deployment process:

On every Git push:

A new Docker image is built from the codebase.

The image is tagged and pushed to AWS ECR.

Deployment to EC2:

The EC2 instance pulls the latest Docker image from ECR.

A container is launched to serve the FastAPI UI for the summarization model, making it accessible to users over the web.

This deployment pipeline ensures a seamless and automated process for pushing updates from development to production in real time.

